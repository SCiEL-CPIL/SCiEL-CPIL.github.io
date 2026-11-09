"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
        
    scores = []
    for op in ready_operations:
        # Get the minimum duration among machine options
        # If machine_options is empty, treat duration as 0 or large penalty
        if op["machine_options"]:
            min_dur = min(op["machine_options"].values())
        else:
            min_dur = 0.0
            
        # Heuristic: Prioritize jobs with more remaining work.
        # Tie-breaker: Prioritize longer operations (LPT) to reduce makespan variance.
        # We scale remaining_work to dominate, but allow min_dur to break ties.
        # Using a large multiplier ensures remaining_work is the primary key.
        score = op["remaining_work"] * 1000.0 + min_dur
        
        scores.append(score)
        
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
