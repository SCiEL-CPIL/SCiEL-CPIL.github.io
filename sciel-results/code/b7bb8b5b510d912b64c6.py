"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Extract metrics for each ready operation
    # We use a hybrid of Most Work Remaining (MWRT) and Shortest Processing Time (SPT)
    # MWRT favors jobs with more total work left, reducing the chance of a long job being delayed.
    # SPT favors operations that take less time, freeing up machines faster.
    
    remaining_works = []
    min_durations = []
    
    for op in ready_operations:
        remaining_works.append(op["remaining_work"])
        
        # Get the minimum duration among available machine options
        # This assumes the scheduler will pick the fastest machine if not specified,
        # or that shorter durations are generally better for makespan.
        machine_opts = op["machine_options"]
        if machine_opts:
            durations = list(machine_opts.values())
            min_dur = min(durations)
        else:
            min_dur = 0
        min_durations.append(min_dur)
        
    # Normalize metrics to [0, 1] range based on current set
    max_rw = max(remaining_works) if remaining_works else 1
    max_min_dur = max(min_durations) if min_durations else 1
    
    # Handle edge cases where max is 0 to avoid division by zero
    if max_rw == 0:
        max_rw = 1
    if max_min_dur == 0:
        max_min_dur = 1
        
    best_index = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        rw = remaining_works[i]
        md = min_durations[i]
        
        norm_rw = rw / max_rw
        norm_md = md / max_min_dur
        
        # Weights: 
        # 0.6 for MWRT (keep long jobs moving)
        # 0.4 for SPT (minimize machine occupation time)
        # Score is to be maximized. We subtract norm_md because shorter is better.
        score = 0.6 * norm_rw - 0.4 * norm_md
        
        if score > best_score:
            best_score = score
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
