"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Dispatch rule using a Machine-Centric Representative Strategy.
    
    1. For each machine that has at least one ready operation, identify the 
       "best" operation to run on it using Shortest Processing Time (SPT).
       This ensures that if a machine is available, we assign it a task that 
       finishes quickly, reducing potential blocking.
    2. Among these selected "representative" operations (one per machine), 
       select the one belonging to the job with the Longest Remaining 
       Processing Time (LRPT). This ensures that critical jobs are prioritized.
    3. Tie-breaker: Prefer the operation with the shorter duration.
    """
    if not ready_operations:
        return 0

    # Step 1: Group operations by machine
    # machine -> list of (index, duration)
    machine_candidates = defaultdict(list)
    
    for idx, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        for machine, duration in machine_options.items():
            machine_candidates[machine].append((idx, duration))
    
    # Step 2: For each machine, pick the best op (SPT: min duration)
    # We store the representative op for each machine
    representatives = []
    
    for machine, ops in machine_candidates.items():
        if not ops:
            continue
        # Find op with min duration on this machine
        # If tie, pick the one with lower index (arbitrary but consistent)
        best_op_idx, best_duration = min(ops, key=lambda x: (x[1], x[0]))
        
        # Retrieve the operation details for scoring in step 3
        op_data = ready_operations[best_op_idx]
        remaining_work = op_data["remaining_work"]
        
        # Store representative: (remaining_work, best_duration, index)
        representatives.append((remaining_work, best_duration, best_op_idx))
    
    if not representatives:
        return 0

    # Step 3: Select the representative with max remaining_work
    # Tie-breaker: min duration
    # Sort by: -remaining_work (desc), duration (asc)
    # We want max remaining_work, so we negate it for min-sort or use max with key
    best_rep = max(representatives, key=lambda x: (x[0], -x[1]))
    
    return int(best_rep[2])
# EVOLVE-BLOCK-END
