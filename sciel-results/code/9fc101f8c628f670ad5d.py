import time as _time

_START_TIME = _time.time()
_DEADLINE = 140.0

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        # Fallback: Max remaining work
        return int(max(range(n), key=lambda i: ready_operations[i]["remaining_work"]))
    
    if n == 1:
        return 0
    
    best_idx = 0
    best_score = -float('inf')
    best_min_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_duration = min(durations) if durations else 0
        
        remaining_work = op["remaining_work"]
        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1
        
        # NSPT-inspired: scale the SPT penalty by the job's remaining work.
        # This means: for jobs with lots of remaining work, we are less willing to
        # penalize a slightly longer operation (since the job is critical and we want
        # to keep it moving), but for jobs with little remaining work, we strongly
        # prefer short operations to free up machines quickly.
        # 
        # The urgency term (remaining_work / remaining_ops) captures the average
        # workload per remaining operation, which is a good proxy for criticality.
        #
        # Combined score: urgency - alpha * (min_duration / remaining_work)
        # But we need to be careful: if remaining_work is 0, we avoid division by zero.
        
        urgency = remaining_work / remaining_ops
        
        # NSPT component: normalize min_duration by remaining_work to create
        # a relative penalty that decreases as the job becomes less critical
        if remaining_work > 0:
            nspt_penalty = min_duration / remaining_work
        else:
            nspt_penalty = min_duration
        
        # Blend: urgency dominates, NSPT refines
        score = urgency - 0.3 * nspt_penalty
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_min_duration = min_duration
        elif score == best_score:
            # Tie-breaker: Prefer shorter processing time
            if min_duration < best_min_duration:
                best_idx = i
                best_min_duration = min_duration
                
    return int(best_idx)
