"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = None

def _ensure_init():
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()

def _time_left():
    if _dispatch_start_time is None:
        return 115.0
    return 115.0 - (_time.time() - _dispatch_start_time)

def dispatch(ready_operations, time):
    _ensure_init()
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If we're running low on time, use the fastest heuristic
    if _time_left() < 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = []
    best_machines = []
    
    for op in ready_operations:
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if not durations:
            scores.append(0.0)
            best_machines.append(None)
            continue
        
        min_dur = min(durations)
        num_machines = len(durations)
        remaining_ops = op.get("remaining_ops", 1)
        
        # Determine best machine for this op: Min duration
        best_machine = None
        min_dur_val = float('inf')
        for m, d in op["machine_options"].items():
            if d < min_dur_val:
                min_dur_val = d
                best_machine = m
        
        # Score Calculation
        # Primary: Most Work Remaining (MWR) - Critical Path
        score = work
        
        # Secondary: Prefer jobs closer to completion (fewer remaining ops)
        # Completing a job reduces active contention
        score -= 0.05 * remaining_ops
        
        # Tertiary: Prefer operations with shorter processing times
        # Keeps machines free and reduces blocking
        score -= 0.01 * min_dur
        
        # Quaternary: Slight bonus for flexibility (more machine options)
        score += 0.001 * num_machines
        
        scores.append(score)
        best_machines.append(best_machine)
    
    # Find the best index
    best_idx = int(np.argmax(scores))
    
    # Tie-breaking: If scores are very close, prefer lower min_duration
    best_score = scores[best_idx]
    tie_indices = [i for i in range(n) if abs(scores[i] - best_score) < 1e-9]
    
    if len(tie_indices) > 1:
        def get_min_dur(idx):
            durs = list(ready_operations[idx]["machine_options"].values())
            return min(durs) if durs else float('inf')
        
        best_idx = min(tie_indices, key=get_min_dur)
    
    # Return index and machine to ensure optimal machine selection
    return (best_idx, best_machines[best_idx])
# EVOLVE-BLOCK-END
