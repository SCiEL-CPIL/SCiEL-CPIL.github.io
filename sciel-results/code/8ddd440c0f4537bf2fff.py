"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # 5s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: 
    1. For each ready operation, find the machine with the minimum processing time.
    2. Score each operation by: (min_duration + remaining_work).
    3. Pick the operation with the highest score.
    4. Return (index, best_machine).
    
    Rationale: 
    - 'remaining_work' prioritizes jobs with more total work left (LPT-like on job size).
    - 'min_duration' prioritizes operations that are long on the best machine, 
      ensuring long tasks are started early to avoid delaying the critical path at the end.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to simple heuristic to avoid timeout
        return 0

    if not ready_operations:
        return 0

    best_idx = 0
    best_score = -1
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Find the best machine (min duration) for this operation
        min_dur = float('inf')
        chosen_machine = None
        for m, dur in op['machine_options'].items():
            if dur < min_dur:
                min_dur = dur
                chosen_machine = m
        
        # If no machine options, skip or handle error (should not happen)
        if chosen_machine is None:
            continue

        # Score: prioritize long current ops and jobs with lots of remaining work
        # Weights can be tuned, but 1:1 is a start.
        score = min_dur + op['remaining_work']
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = chosen_machine

    # If no valid operation was found (shouldn't happen if ready_operations is not empty), return 0
    if best_machine is None:
        return 0
        
    return (best_idx, best_machine)

# EVOLVE-BLOCK-END
