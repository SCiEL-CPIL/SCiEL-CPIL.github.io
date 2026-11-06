"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy:
    1. Prioritize operations with fewest machine options (constraint tightness) to prevent starvation.
    2. Break ties by remaining work (more work first to balance load).
    3. Select the machine with minimum duration for the chosen operation.
    
    Returns:
        index of the ready operation to start, or (index, machine) tuple.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute constraint tightness: fewer machine options = higher priority
    # Use negative count so that fewer options = higher score
    scores = np.empty(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        machine_opts = op.get("machine_options", {})
        num_options = len(machine_opts) if machine_opts else 1
        
        # Constraint tightness: inverse of number of options
        # Fewer options -> higher tightness -> higher priority
        tightness = 1.0 / max(num_options, 1)
        
        # Remaining work as secondary criterion
        remaining_work = op.get("remaining_work", 0)
        
        # Combine: primary is tightness, secondary is remaining work
        # Scale remaining work to be secondary
        max_remaining = 1.0
        for j in range(n):
            rw = ready_operations[j].get("remaining_work", 0)
            if rw > max_remaining:
                max_remaining = rw
        
        if max_remaining > 0:
            work_normalized = remaining_work / max_remaining
        else:
            work_normalized = 0.0
        
        # Weight tightness heavily, work as tiebreaker
        scores[i] = tightness * 100.0 + work_normalized
    
    # Pick the operation with highest score
    best_idx = int(np.argmax(scores))
    
    # For the chosen operation, select the machine with minimum duration
    chosen_op = ready_operations[best_idx]
    machine_opts = chosen_op.get("machine_options", {})
    
    if machine_opts:
        # Select machine with minimum processing time
        best_machine = min(machine_opts.keys(), key=lambda m: machine_opts[m])
        return (best_idx, best_machine)
    else:
        return best_idx
# EVOLVE-BLOCK-END
