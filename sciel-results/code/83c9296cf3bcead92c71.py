"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy: Combine job criticality (remaining work), shortest processing time
    on the best machine, and machine load balancing to minimize makespan.
    
    Returns: (index, machine) tuple where index is the ready operation to start
             and machine is the selected machine from its machine_options.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with shortest duration for this single op
        best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        return (0, best_machine)
    
    scores = np.zeros(n)
    selected_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Find the best machine for this operation (shortest processing time)
        best_machine = min(machine_options.items(), key=lambda x: x[1])
        best_duration = best_machine[1]
        selected_machines[i] = best_machine[0]
        
        # Criticality: how much work remains for this job
        # Higher remaining work -> more urgent
        criticality = remaining_work
        
        # Processing time factor: shorter is better (SPT-like)
        # Use inverse so shorter processing time gets higher score
        spt_factor = 1.0 / (best_duration + 1e-9)
        
        # Urgency from remaining operations count
        # More remaining ops means this job has more to do, so it's more critical
        urgency = remaining_ops
        
        # Combined score: weight criticality heavily, then SPT, then urgency
        # This prioritizes jobs with most remaining work, preferring those
        # that can be processed quickly, and breaking ties by remaining ops
        scores[i] = (
            10.0 * criticality +      # Primary: job criticality
            1.0 * spt_factor +          # Secondary: shortest processing time
            0.1 * urgency               # Tertiary: urgency
        )
    
    # Select the operation with highest combined score
    best_idx = int(np.argmax(scores))
    
    return (best_idx, selected_machines[best_idx])
# EVOLVE-BLOCK-END
