"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # For each ready operation, compute a priority score
    # that balances job urgency (remaining_work) and operation impact
    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Find the minimum duration across machine options (best machine for this op)
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 0.0
        
        # Composite score: emphasize remaining work but also consider op duration
        # Jobs with more remaining work are more critical; longer ops have more impact
        score = 0.7 * remaining_work + 0.3 * min_duration
        scores.append(score)
    
    # Return index of operation with highest priority score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
