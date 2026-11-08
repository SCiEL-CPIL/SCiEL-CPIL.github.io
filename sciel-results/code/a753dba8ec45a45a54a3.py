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
    
    best_idx = 0
    best_score = float('-inf')
    best_machine = None
    
    n = len(ready_operations)
    
    for i, op in enumerate(ready_operations):
        options = op['machine_options']
        if not options:
            continue
        
        # Find the machine with minimum duration for this operation
        min_dur = min(options.values())
        best_machine_for_op = next(m for m, d in options.items() if d == min_dur)
        
        remaining_work = op['remaining_work']
        
        # Heuristic: WSPT-like with time-awareness
        # Prioritize operations with more remaining work relative to processing time
        # Also consider current time to avoid long delays
        if remaining_work <= 0:
            score = 0
        else:
            # WSPT: remaining_work / processing_time
            # Add a small time penalty to favor earlier completion
            score = remaining_work / (min_dur + 1)
            # Slight bonus for operations that can finish sooner (shorter min_dur)
            score += 0.1 / (min_dur + 1)
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
            
    return (best_idx, best_machine)
