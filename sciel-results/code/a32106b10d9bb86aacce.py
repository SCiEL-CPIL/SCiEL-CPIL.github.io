"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Most Constrained First (MCF) with Work Weighting.
    
    Prioritizes operations that have a high amount of remaining work relative to
    their flexibility (number of available machine options). This ensures that
    critical, inflexible operations are scheduled before flexible ones to avoid
    bottlenecks and deadlocks.
    
    Machine selection: For the chosen operation, select the machine with the
    minimum processing time to minimize local completion time.
    """
    if not ready_operations:
        return 0

    best_index = 0
    best_score = -float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue

        # Calculate flexibility: number of available machines
        num_options = len(machine_options)
        
        # Find the machine with the minimum duration for this operation
        # This is the best machine to assign to this op to minimize its completion time
        best_m, min_duration = min(machine_options.items(), key=lambda item: item[1])
        
        remaining_work = op["remaining_work"]
        
        # Score: High remaining work and low flexibility (few options) is better.
        # This is a "Constraint Propagation" style heuristic.
        # We add 1e-9 to avoid division by zero (though num_options >= 1).
        score = remaining_work / (num_options + 1e-9)
        
        # We want to maximize this score
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m

    # Handle edge case where no valid machine was found
    if best_machine is None:
        return 0
        
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
