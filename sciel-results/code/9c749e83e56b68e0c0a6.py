"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using a Ratio Heuristic:
    Score = Remaining Critical Path Work / Minimum Operation Duration
    
    This approach favors operations that have a long remaining critical path
    (high remaining_work) but a short current processing time (low min_duration).
    This encourages starting jobs with many subsequent steps early to maximize
    parallelism and pipeline utilization, while deprioritizing long single-step
    operations that might bottleneck specific machines if started too late.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    best_score = -1e18
    best_index = 0
    
    for i in range(n):
        op = ready_operations[i]
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Calculate minimum duration across available machines
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 0.0
            
        # Prevent division by zero
        if min_duration < 1e-9:
            min_duration = 1e-9
            
        # Calculate ratio score
        # Higher score indicates higher priority
        score = remaining_work / min_duration
        
        # Tie-breaker: If scores are extremely close, prefer the one with
        # more remaining operations (depth) to ensure robustness
        # We add a tiny fraction of remaining_ops to break ties
        remaining_ops = op["remaining_ops"]
        # Scale remaining_ops down so it only affects near-ties
        tie_breaker = remaining_ops * 1e-6
        
        total_score = score + tie_breaker
        
        if total_score > best_score:
            best_score = total_score
            best_index = i
            
    return best_index
# EVOLVE-BLOCK-END
