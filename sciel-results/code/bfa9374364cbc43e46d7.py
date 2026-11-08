"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Deadline guard
_START_TIME = time_module.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _elapsed_ok():
    return (time_module.time() - _START_TIME) < (_TIME_LIMIT - _SAFETY_MARGIN)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Check deadline
    if not _elapsed_ok():
        # Fallback to simple index to avoid timeout
        return 0

    # Extract durations. 
    # machine_options is {machine: duration}. 
    # We assume the operation is assigned to one of these. 
    # If multiple, we might take the max or min? 
    # In open shop, usually one machine per op. 
    # Let's take the max duration available as a proxy for "heaviness" if flexible, 
    # or just the single value if fixed.
    
    scores = []
    for op in ready_operations:
        options = op.get("machine_options", {})
        if options:
            # If multiple machines, picking the one with the longest duration 
            # might be bad if we can choose a shorter one. 
            # But we don't return the machine. 
            # So we assume the "cost" of this operation is its duration.
            # If we can't choose the machine, we can't optimize for it here.
            # Let's assume the duration is the value. 
            # If there are multiple, maybe the simulator picks the best one? 
            # Or maybe we should use the max as a conservative estimate of delay?
            # Let's use the max duration to be safe (worst case delay).
            duration = max(options.values())
        else:
            duration = 0
            
        remaining_work = op.get("remaining_work", 0)
        
        # Heuristic: 
        # 1. Longest Operation First (LPT) is strong for makespan.
        # 2. Job with most remaining work (Tail) is also strong.
        # Let's combine them. 
        # A simple effective rule is to prioritize by duration, 
        # but breaking ties or adding a small weight to remaining work.
        
        # Let's try: Score = duration + 0.1 * remaining_work
        # Or just duration.
        # The seed used remaining_work. 
        # Let's try duration first.
        
        scores.append(duration)
        
    if not scores:
        return 0
        
    # If all durations are 0, fallback to remaining work
    if max(scores) == 0:
         scores = [op.get("remaining_work", 0) for op in ready_operations]
         if max(scores) == 0:
             return 0
             
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
