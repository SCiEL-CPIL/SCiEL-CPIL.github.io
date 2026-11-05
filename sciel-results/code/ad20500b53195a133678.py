"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Select the ready operation that minimizes a combined score
    of weighted remaining work and machine load.
    
    We use a composite heuristic:
    1. For each ready operation, compute the minimum possible completion time
       if started now (current time + min duration across available machines).
    2. Prioritize operations with higher remaining work (critical path consideration).
    3. Break ties by preferring operations on less-loaded machines (estimated by
       average duration of remaining ops on that machine).
    
    This approach balances urgency (remaining work) with resource contention.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Precompute scores for each ready operation
    scores = []
    for op in ready_operations:
        job = op["job"]
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Minimum duration to complete this operation (best case)
        min_duration = min(machine_options.values()) if machine_options else 0
        max_duration = max(machine_options.values()) if machine_options else 0
        
        # Estimated completion time if started now
        est_completion = time + min_duration
        
        # Weight remaining work heavily - operations with more remaining work
        # are on the critical path and should be prioritized
        work_score = remaining_work
        
        # Prefer operations that can be completed sooner (lower est_completion)
        time_score = est_completion
        
        # Combine: prioritize high remaining work, break ties by earlier completion
        # We want to maximize work_score but minimize time_score
        # Normalize to avoid scale issues
        score = work_score * 1000.0 - time_score
        
        # Small bonus for operations with fewer remaining ops (closer to job completion)
        # This helps finish jobs earlier, reducing makespan
        if remaining_ops == 1:
            score += 500.0
        
        scores.append(score)
    
    # Select the operation with the highest composite score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
