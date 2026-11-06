"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = None


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses a composite score combining:
    - Minimum processing time (prefer shorter ops to finish quickly)
    - Job urgency (prefer jobs with fewer remaining ops)
    - Machine contention avoidance (prefer ops on less-loaded machines)
    """
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Safety: if we're close to timeout, return quickly
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > 140.0:
        # Fallback: just pick first
        return 0
    
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]
        
        # Get processing times for each machine option
        durations = list(machine_options.values())
        if not durations:
            min_dur = 1e9
            mean_dur = 1e9
        else:
            min_dur = min(durations)
            mean_dur = sum(durations) / len(durations)
        
        # Component 1: Prefer shorter minimum processing time
        # Normalize by making shorter times have higher score
        time_score = 1.0 / (min_dur + 1e-6)
        
        # Component 2: Prefer jobs with fewer remaining operations (urgency)
        # Jobs closer to completion should be prioritized
        urgency_score = 1.0 / (remaining_ops + 1e-6)
        
        # Component 3: Machine contention - prefer operations where the best machine
        # has relatively low load compared to other options
        # If min_dur is much smaller than mean_dur, there's a clear best machine
        if len(durations) > 1:
            spread = (mean_dur - min_dur) / (mean_dur + 1e-6)
        else:
            spread = 0.0
        # Higher spread means more flexibility, which is good
        flexibility_score = 1.0 + spread
        
        # Component 4: Remaining work - less remaining work means job is closer to done
        work_score = 1.0 / (remaining_work + 1e-6)
        
        # Composite score with weights
        # Weight urgency and time heavily, flexibility and work moderately
        score = (
            2.0 * time_score +
            1.5 * urgency_score +
            0.5 * flexibility_score +
            0.5 * work_score
        )
        
        scores.append(score)
    
    # Return index of operation with highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
