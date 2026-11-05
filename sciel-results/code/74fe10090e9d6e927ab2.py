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
        
        # Combined score: Prioritize jobs with most remaining work (MWRT)
        # but penalize operations with long processing times (SPT component).
        # Alpha = 0.5 balances the two effects.
        # This avoids the starvation of long jobs (pure SPT) and the blocking
        # of machines by long ops (pure MWRT).
        score = remaining_work - 0.5 * min_duration
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_min_duration = min_duration
        elif score == best_score:
            # Tie-breaker: Prefer shorter processing time to free machines sooner
            if min_duration < best_min_duration:
                best_idx = i
                best_min_duration = min_duration
                
    return int(best_idx)
