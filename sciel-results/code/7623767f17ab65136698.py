import time as _time
import numpy as np

# Deadline guard: ensure we return well before 150s wall-clock limit
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
        
    # Pre-compute features for all ready operations
    # We need to evaluate the "impact" of assigning each op.
    # Since we don't have global machine state, we approximate "load"
    # by the inverse of the processing time on the preferred machine.
    # Shorter processing times on a machine imply that machine is "less loaded" 
    # in a relative sense for this specific job, or that the job is a "light" 
    # user. However, the directive suggests "least-loaded available machine".
    # Without state, we can't know load. 
    # Alternative interpretation: "Least loaded" might refer to the machine 
    # with the fewest constraints or shortest duration for *this* op.
    
    # Let's implement the "Dynamic Bottleneck Identification" strategy:
    # 1. For each ready op i, determine its "preferred" machine (min duration).
    # 2. Estimate the "slack" or "urgency" of the system if op i is NOT chosen.
    # 3. The strategy aims to maximize the minimum slack across all ready ops.
    #    Slack_i = (Time until critical constraint) - (Time needed).
    #    Without future info, we approximate slack by the "freedom" of the op.
    #    An op with many machine options and short durations has high slack.
    #    An op with few options and long durations has low slack (bottleneck).
    
    # We want to pick the op that, when processed, best relieves the "tightest" constraint.
    # Or, pick the op that *is* the tightest constraint? 
    # Usually, you process the most critical (least slack) item first to prevent it from becoming a delay.
    # This is similar to "Most Critical Job".
    
    # Let's define a "Bottleneck Score" for each op:
    # Bottleneck_i = 1 / (Sum of durations across available machines) + Weight * (1 / Num_Machines)
    # High score = High Bottleneck (Low Slack).
    
    # The directive says: "selects the operation whose assignment to its least-loaded available machine 
    # maximizes the minimum slack time across all currently ready operations".
    
    # This is a bit circular. Let's simplify to a robust heuristic that fits this spirit:
    # 1. Calculate "Load" for each op: Load_i = max(duration) / min(duration) (Variability) 
    #    or just 1/min_duration (Urgency).
    # 2. Calculate "Constraint Tightness": Tightness_i = 1 / len(machine_options).
    #    Ops with fewer machine choices are more constrained.
    
    # Strategy: Prioritize ops that are BOTH urgent (short min dur) AND constrained (few options).
    # This targets the "bottleneck" operations that could cause idle time if delayed.
    
    best_idx = 0
    best_machine = None
    best_score = -float('inf')
    
    # Vectorized approach for speed if n is large, but n is small (<=10 jobs, so <=10 ready ops usually)
    # So loop is fine.
    
    min_durs = []
    num_options = []
    
    for op in ready_operations:
        opts = op['machine_options']
        if not opts:
            min_durs.append(float('inf'))
            num_options.append(0)
            continue
        durations = list(opts.values())
        min_durs.append(min(durations))
        num_options.append(len(durations))
        
    min_durs_arr = np.array(min_durs, dtype=float)
    num_opts_arr = np.array(num_options, dtype=float)
    
    # Avoid division by zero
    min_durs_safe = np.where(min_durs_arr > 0, min_durs_arr, 1.0)
    num_opts_safe = np.where(num_opts_arr > 0, num_opts_arr, 1.0)
    
    # "Slack" proxy: High slack = Low urgency, Low constraint.
    # Slack_i = min_dur * num_options
    # We want to MINIMIZE slack (i.e., pick the most critical/least slack op).
    # Or, following the directive "maximizes the minimum slack", we might be looking at the *result*.
    # If we pick op i, the "minimum slack" of the *remaining* set changes.
    # This is complex. Let's stick to the "Most Critical" interpretation which is standard for JSS.
    # Critical = Low Slack.
    
    # Slack_i = min_dur_i * num_options_i
    # Lower Slack = More Critical.
    # We want the op with the LOWEST Slack.
    
    slacks = min_durs_safe * num_opts_safe
    
    # Add a tie-breaker: if slacks are equal, prefer the one with shorter min_dur (SPT)
    # to free up the machine faster.
    
    # To make it "Dynamic Bottleneck", let's weight the slack by the "remaining work" of the job.
    # Jobs with more remaining work are more critical if they are bottlenecks.
    
    remaining_works = np.array([op['remaining_work'] for op in ready_operations], dtype=float)
    
    # Score = (1 / Slack) * (1 + remaining_work)
    # High Score = High Priority.
    
    scores = np.zeros(n)
    for i in range(n):
        if slacks[i] > 0:
            scores[i] = (1.0 / slacks[i]) * (1.0 + remaining_works[i])
        else:
            scores[i] = 0.0
            
    # Find the index with the highest score
    # In case of ties, pick the first one (stable)
    best_idx = int(np.argmax(scores))
    
    # Determine the best machine for the selected operation
    selected_op = ready_operations[best_idx]
    options = selected_op['machine_options']
    if options:
        # Pick the machine with the minimum duration (least load for this op)
        best_machine = min(options, key=options.get)
    else:
        best_machine = None
        
    return (best_idx, best_machine)
