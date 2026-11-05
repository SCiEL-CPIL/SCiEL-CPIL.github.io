import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = 140.0

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        # Fallback: Max remaining work, tie-break min duration
        best_idx = 0
        best_rw = -1
        best_md = float('inf')
        for i in range(n):
            rw = ready_operations[i]["remaining_work"]
            mds = list(ready_operations[i]["machine_options"].values())
            md = min(mds) if mds else 0
            if rw > best_rw or (rw == best_rw and md < best_md):
                best_rw = rw
                best_md = md
                best_idx = i
        return best_idx

    best_idx = 0
    best_score = -float('inf')
    best_min_duration = float('inf')
    best_remaining_work = -1
    best_remaining_ops = -1

    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_duration = min(durations) if durations else 0
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Heuristic: Enhanced WSPT with dynamic urgency
        # Base score: WSPT (Weighted Shortest Processing Time)
        # Weight: Remaining work of the job
        # Time: Minimum possible duration of the current operation
        
        # Avoid division by zero
        effective_duration = max(min_duration, 1e-6)
        
        # Base WSPT score
        wspt_score = remaining_work / effective_duration
        
        # Dynamic urgency component: penalize operations from jobs with 
        # high work-per-remaining-op ratio (critical jobs)
        # This accounts for both total remaining work and number of remaining operations
        if remaining_ops > 0:
            work_per_op = remaining_work / remaining_ops
            # Normalize to create a penalty/bonus that emphasizes urgency
            # Jobs with high work_per_op are more critical and should be prioritized
            urgency_factor = work_per_op / max(remaining_work, 1e-6)
            # Scale the urgency contribution
            urgency_score = 0.1 * urgency_factor
        else:
            urgency_score = 0.0
        
        # Combined score: WSPT + urgency adjustment
        score = wspt_score + urgency_score
        
        # Secondary consideration: If scores are very close, prefer shorter duration
        # This helps in reducing idle time on machines.
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_min_duration = min_duration
            best_remaining_work = remaining_work
            best_remaining_ops = remaining_ops
        elif abs(score - best_score) < 1e-9:
            # Tie-breaker 1: Prefer shorter processing time
            if min_duration < best_min_duration:
                best_idx = i
                best_min_duration = min_duration
                best_remaining_work = remaining_work
                best_remaining_ops = remaining_ops
            # Tie-breaker 2: If duration is same, prefer higher remaining work
            elif min_duration == best_min_duration and remaining_work > best_remaining_work:
                best_idx = i
                best_min_duration = min_duration
                best_remaining_work = remaining_work
                best_remaining_ops = remaining_ops
            # Tie-breaker 3: If both same, prefer fewer remaining ops (closer to completion)
            elif min_duration == best_min_duration and remaining_work == best_remaining_work and remaining_ops < best_remaining_ops:
                best_idx = i
                best_min_duration = min_duration
                best_remaining_work = remaining_work
                best_remaining_ops = remaining_ops

    return best_idx
