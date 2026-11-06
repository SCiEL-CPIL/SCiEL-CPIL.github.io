"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Uses a Pair-Based Greedy Heuristic with Potential Contention:
    1. Calculate 'potential contention' for each machine: the number of ready operations
       that can be processed on that machine.
    2. For each ready operation and each eligible machine, calculate a score:
       Score = remaining_work / (duration * (1 + potential_contention))
    3. This prioritizes operations with high remaining work (urgency) that can be
       processed quickly on machines with low potential congestion.
    4. Select the operation-machine pair with the highest score.
    
    Returns: (index, machine) tuple.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Step 1: Calculate potential contention for each machine
    # potential_contention[m] = number of ready ops that can use machine m
    potential_contention = {}
    for op in ready_operations:
        for m in op["machine_options"].keys():
            potential_contention[m] = potential_contention.get(m, 0) + 1
            
    # Step 2: Evaluate all (operation, machine) pairs
    best_score = -1.0
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        if rw <= 0:
            continue
            
        for m, d in op["machine_options"].items():
            if d <= 0:
                continue
                
            contention = potential_contention.get(m, 0)
            # Effective cost: duration penalized by potential congestion
            # Using (1 + contention) to ensure the base duration is always considered
            eff_cost = d * (1 + contention)
            
            # Score: Urgency / Cost
            score = rw / eff_cost
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m
                
    # Fallback if no valid pair found
    if best_machine is None:
        # Pick the first operation and its first machine
        if ready_operations:
            best_idx = 0
            machine_options = ready_operations[0]["machine_options"]
            if machine_options:
                best_machine = list(machine_options.keys())[0]
            else:
                best_machine = 0
        else:
            return 0
            
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
