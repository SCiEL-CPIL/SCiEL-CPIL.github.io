"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        rw = op["remaining_work"]
        # Get the minimum duration among machine options
        # If only one, it's that value.
        durations = list(op["machine_options"].values())
        if not durations:
            d_min = 0
        else:
            d_min = min(durations)
        
        # Composite score: Weighted sum of Remaining Work and Immediate Op Duration
        # LPT logic: Larger is better.
        score = rw + 0.5 * d_min
        scores.append(score)
        
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
