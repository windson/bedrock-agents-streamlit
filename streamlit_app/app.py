import invoke_agent as agenthelper
import streamlit as st
import json
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

# Streamlit page configuration
st.set_page_config(page_title="Co. Portfolio Creator", page_icon=":robot_face:", layout="wide")

# Function to crop image into a circle
def crop_to_circle(image):
    mask = Image.new('L', image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0) + image.size, fill=255)
    result = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
    result.putalpha(mask)
    return result

def process_routing_trace(event, step, _sub_agent_name, _time_before_routing=None):
    """Process routing classifier trace events."""
   
    _route = event['trace']['trace']['routingClassifierTrace']
    
    if 'modelInvocationInput' in _route:
        #print("Processing modelInvocationInput")
        container = st.container(border=True)                            
        container.markdown(f"""**Choosing a collaborator for this request...**""")
        return datetime.datetime.now(), step, _sub_agent_name, None, None
        
    if 'modelInvocationOutput' in _route and _time_before_routing:
        #print("Processing modelInvocationOutput")
        _llm_usage = _route['modelInvocationOutput']['metadata']['usage']
        inputTokens = _llm_usage['inputTokens']
        outputTokens = _llm_usage['outputTokens']
        
        _route_duration = datetime.datetime.now() - _time_before_routing

        _raw_resp_str = _route['modelInvocationOutput']['rawResponse']['content']
        _raw_resp = json.loads(_raw_resp_str)
        _classification = _raw_resp['content'][0]['text'].replace('<a>', '').replace('</a>', '')

        if _classification == "undecidable":
            text = f"No matching collaborator. Revert to 'SUPERVISOR' mode for this request."
        elif _classification in (_sub_agent_name, 'keep_previous_agent'):
            step = math.floor(step + 1)
            text = f"Continue conversation with previous collaborator"
        else:
            _sub_agent_name = _classification
            step = math.floor(step + 1)
            text = f"Use collaborator: '{_sub_agent_name}'"

        time_text = f"Intent classifier took {_route_duration.total_seconds():,.1f}s"
        container = st.container(border=True)                            
        container.write(text)
        container.write(time_text)
        
        return step, _sub_agent_name, inputTokens, outputTokens

def process_orchestration_trace(event, agentClient, step):
    """Process orchestration trace events."""
    _orch = event['trace']['trace']['orchestrationTrace']
    inputTokens = 0
    outputTokens = 0
    
    if "invocationInput" in _orch:
        _input = _orch['invocationInput']
        
        if 'knowledgeBaseLookupInput' in _input:
            with st.expander("Using knowledge base", False, icon=":material/plumbing:"):
                st.write("knowledge base id: " + _input["knowledgeBaseLookupInput"]["knowledgeBaseId"])
                st.write("query: " + _input["knowledgeBaseLookupInput"]["text"].replace('$', '\$'))
                
        if "actionGroupInvocationInput" in _input:
            function = _input["actionGroupInvocationInput"]["function"]
            with st.expander(f"Invoking Tool - {function}", False, icon=":material/plumbing:"):
                st.write("function : " + function)
                st.write("type: " + _input["actionGroupInvocationInput"]["executionType"])
                if 'parameters' in _input["actionGroupInvocationInput"]:
                    st.write("*Parameters*")
                    params = _input["actionGroupInvocationInput"]["parameters"]
                    st.table({
                        'Parameter Name': [p["name"] for p in params],
                        'Parameter Value': [p["value"] for p in params]
                    })

        if 'codeInterpreterInvocationInput' in _input:
            with st.expander("Code interpreter tool usage", False, icon=":material/psychology:"):
                gen_code = _input['codeInterpreterInvocationInput']['code']
                st.code(gen_code, language="python")
                    
    if "modelInvocationOutput" in _orch:
        if "usage" in _orch["modelInvocationOutput"]["metadata"]:
            inputTokens = _orch["modelInvocationOutput"]["metadata"]["usage"]["inputTokens"]
            outputTokens = _orch["modelInvocationOutput"]["metadata"]["usage"]["outputTokens"]
                    
    if "rationale" in _orch:
        if "agentId" in event["trace"]:
            agentData = agentClient.get_agent(agentId=event["trace"]["agentId"])
            agentName = agentData["agent"]["agentName"]
            chain = event["trace"]["callerChain"]
            
            container = st.container(border=True)
            
            if len(chain) <= 1:
                step = math.floor(step + 1)
                container.markdown(f"""#### Step  :blue[{round(step,2)}]""")
            else:
                step = step + 0.1
                container.markdown(f"""###### Step {round(step,2)} Sub-Agent  :red[{agentName}]""")
            
            container.write(_orch["rationale"]["text"].replace('$', '\$'))

    if "observation" in _orch:
        _obs = _orch['observation']
        
        if 'knowledgeBaseLookupOutput' in _obs:
            with st.expander("Knowledge Base Response", False, icon=":material/psychology:"):
                _refs = _obs['knowledgeBaseLookupOutput']['retrievedReferences']
                _ref_count = len(_refs)
                st.write(f"{_ref_count} references")
                for i, _ref in enumerate(_refs, 1):
                    st.write(f"  ({i}) {_ref['content']['text'][0:200]}...")

        if 'actionGroupInvocationOutput' in _obs:
            with st.expander("Tool Response", False, icon=":material/psychology:"):
                st.write(_obs['actionGroupInvocationOutput']['text'].replace('$', '\$'))

        if 'codeInterpreterInvocationOutput' in _obs:
            with st.expander("Code interpreter tool usage", False, icon=":material/psychology:"):
                if 'executionOutput' in _obs['codeInterpreterInvocationOutput']:
                    raw_output = _obs['codeInterpreterInvocationOutput']['executionOutput']
                    st.code(raw_output)

                if 'executionError' in _obs['codeInterpreterInvocationOutput']:
                    error_text = _obs['codeInterpreterInvocationOutput']['executionError']
                    st.write(f"Code interpretation error: {error_text}")

                if 'files' in _obs['codeInterpreterInvocationOutput']:
                    files_generated = _obs['codeInterpreterInvocationOutput']['files']
                    st.write(f"Code interpretation files generated:\n{files_generated}")

        if 'finalResponse' in _obs:
            with st.expander("Agent Response", False, icon=":material/psychology:"):
                st.write(_obs['finalResponse']['text'].replace('$', '\$'))
            
    return step, inputTokens, outputTokens


def format_trace_content(trace_data):
    try:
        if isinstance(trace_data, str) and (trace_data.strip().startswith('{') or trace_data.strip().startswith('[')):
            parsed_json = json.loads(trace_data)
            return f"```json\n{json.dumps(parsed_json, indent=2)}\n```"
        elif isinstance(trace_data, (dict, list)):
            return f"```json\n{json.dumps(trace_data, indent=2)}\n```"
        else:
            return str(trace_data).encode('utf-8', errors='ignore').decode('utf-8')
    except:
        return str(trace_data).encode('utf-8', errors='ignore').decode('utf-8')

# Title
st.title("Using multiple agents for scalable generative AI applications")

# Display a text box for input
prompt = st.text_input("Please enter your query?", max_chars=2000)
prompt = prompt.strip()

# Display a primary button for submission
submit_button = st.button("Submit", type="primary")

# Display a button to end the session
end_session_button = st.button("End Session")

# Sidebar for user input
st.sidebar.title("Trace Data")

# Session State Management
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Function to parse and format response
def format_response(response_body):
    try:
        # Try to load the response as JSON
        data = json.loads(response_body)
        # If it's a list, convert it to a DataFrame for better visualization
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return response_body
    except json.JSONDecodeError:
        # If response is not JSON, return as is
        return response_body

# Handling user input and responses
if submit_button and prompt:
    event = {
        "sessionId": "MYSESSION",
        "question": prompt
    }
    response = agenthelper.lambda_handler(event, None)
    
    try:
        # Parse the JSON string
        if response and 'body' in response and response['body']:
            response_data = json.loads(response['body'])
            print("TRACE & RESPONSE DATA ->  ", response_data)
        else:
            print("Invalid or empty response received")
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        response_data = None 
    
    try:
        # Extract the response and trace data
        all_data = format_response(response_data['response'])
        the_response = response_data['trace_data']
    except:
        all_data = "..." 
        the_response = "Apologies, but an error occurred. Please rerun the application" 

    ### START TRACE LOG and OUTPUTS
    # Initialize session state for traces
    if 'traces' not in st.session_state:
        st.session_state['traces'] = []
        st.session_state['step'] = 1
        st.session_state['sub_agent_name'] = None
    
    # When processing a new trace
    if all_data:
        try:
            event = json.loads(all_data)
            
            if 'trace' in event:
                if 'routingClassifierTrace' in event['trace']['trace']:
                    time_before_routing = datetime.datetime.now()
                    result = process_routing_trace(
                        event, 
                        st.session_state['step'], 
                        st.session_state['sub_agent_name'], 
                        time_before_routing
                    )
                    if result:
                        st.session_state['step'], st.session_state['sub_agent_name'], inputTokens, outputTokens = result
                
                elif 'orchestrationTrace' in event['trace']['trace']:
                    result = process_orchestration_trace(
                        event, 
                        agentClient, 
                        st.session_state['step']
                    )
                    if result:
                        st.session_state['step'], inputTokens, outputTokens = result
    
            # Add new trace to session state
            if all_data and (not st.session_state['traces'] or all_data != st.session_state['traces'][0]):
                st.session_state['traces'].insert(0, all_data)
        except json.JSONDecodeError:
            # If all_data is not valid JSON, just store it as is
            if all_data and (not st.session_state['traces'] or all_data != st.session_state['traces'][0]):
                st.session_state['traces'].insert(0, all_data)
    
    # Keep the history and trace_data updates
    st.session_state['history'].append({"question": prompt, "answer": the_response})
    st.session_state['trace_data'] = the_response
    
    # Display traces in sidebar
    for idx, trace in enumerate(st.session_state['traces']):
        with st.sidebar.expander(f"Trace {idx+1}", expanded=(idx==0)):
            formatted_trace = format_trace_content(trace)
            st.markdown(
                f'<div style="min-height: 100px; font-family: monospace; white-space: pre-wrap;">{formatted_trace}</div>', 
                unsafe_allow_html=True
            )
    ### END TRACE LOG and OUTPUTS
  

if end_session_button:
    st.session_state['history'].append({"question": "Session Ended", "answer": "Thank you for using AnyCompany Support Agent!"})
    event = {
        "sessionId": "MYSESSION",
        "question": "placeholder to end session",
        "endSession": True
    }
    agenthelper.lambda_handler(event, None)
    st.session_state['history'].clear()

# Display conversation history
st.write("## Conversation History")

# Load images outside the loop to optimize performance
human_image = Image.open('/home/ubuntu/app/streamlit_app/human_face.png')
robot_image = Image.open('/home/ubuntu/app/streamlit_app/robot_face.jpg')
circular_human_image = crop_to_circle(human_image)
circular_robot_image = crop_to_circle(robot_image)

for index, chat in enumerate(reversed(st.session_state['history'])):
    # Creating columns for Question
    col1_q, col2_q = st.columns([2, 10])
    with col1_q:
        st.image(circular_human_image, width=125)
    with col2_q:
        # Generate a unique key for each question text area
        st.text_area("Q:", value=chat["question"], height=68, key=f"question_{index}", disabled=True)

    # Creating columns for Answer
    col1_a, col2_a = st.columns([2, 10])
    if isinstance(chat["answer"], pd.DataFrame):
        with col1_a:
            st.image(circular_robot_image, width=100)
        with col2_a:
            # Generate a unique key for each answer dataframe
            st.dataframe(chat["answer"], key=f"answer_df_{index}")
    else:
        with col1_a:
            st.image(circular_robot_image, width=150)
        with col2_a:
            # Generate a unique key for each answer text area
            #st.text_area("A:", value=chat["answer"], height=100, key=f"answer_{index}")
            with st.expander("A:", expanded=True):
                st.markdown(
                    f'<div style="min-height: 100px;">{chat["answer"]}</div>', 
                    unsafe_allow_html=True
                )

# Example Prompts Section
st.write("## OCTANK INC. Leave Policy Knowledge Base Prompts")

# Creating a list of prompts for the Knowledge Base section
knowledge_base_prompts = [
    {"Prompt": "How many causal leaves can be availed in an year?"},
    {"Prompt": "Can we apply 4 sick leaves in a row?"},
    {"Prompt": "Help me with different types of leaves at OCTANK INC."},
]

# Displaying the Knowledge Base prompts as a table
st.table(knowledge_base_prompts)

# Test Action Group Prompts
st.write("## OCTANK INC. Leave Action Group Prompts")

# Creating a list of prompts for the Action Group section
action_group_prompts = [
    {"Prompt": "My Emp ID is 1001 and I want to apply causal leave next monday for 2 days."},
    {"Prompt": "Help me with my leave balance. My empid is 1001"},
    {"Prompt": "Can you resend email notification to approver for my leave id: xxx?"}
]

# Displaying the Action Group prompts as a table
st.table(action_group_prompts)

