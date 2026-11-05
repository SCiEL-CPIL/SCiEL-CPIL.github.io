"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using Shortest Processing Time (SPT) heuristic.
    
    This approach prioritizes operations with the shortest processing time on 
    their optimal machine. This helps reduce congestion and allows machines 
    to clear tasks quickly, potentially improving overall makespan by keeping 
    the system responsive. Tie-breakers favor jobs with more remaining 
    operations to ensure long-tail jobs are not delayed.
    
    Returns (index, machine) to explicitly select the machine with the 
    shortest duration for the chosen operation.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        op = ready_operations[0]
        # Select the machine with the shortest duration for this single op
        best_machine = min(op["machine_options"], key=op["machine_options"].get)
        return (0, best_machine)
    
    n = len(ready_operations)
    
    # Calculate metrics for each operation
    candidates = []
    for i in range(n):
        op = ready_operations[i]
        machine_options = op["machine_options"]
        
        if not machine_options:
            # Fallback if no machine options (shouldn't happen)
            min_duration = float('inf')
            best_machine = None
        else:
            # Find machine with minimum duration
            best_machine, min_duration = min(machine_options.items(), key=lambda x: x[1])
            
        remaining_ops = op["remaining_ops"]
        
        # Primary key: min_duration (lower is better)
        # Secondary key: -remaining_ops (higher remaining ops is better, so we negate for sorting)
        # Tertiary key: job id (for stability)
        candidates.append((min_duration, -remaining_ops, op["job"], i, best_machine))
    
    # Sort by min_duration asc, then remaining_ops desc, then job
    candidates.sort()
    
    best_candidate = candidates[0]
    best_index = best_candidate[3]
    best_machine = best_candidate[4]
    
    # If we couldn't find a valid machine (should not happen), fall back to index only
    if best_machine is None:
        return best_index
        
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
