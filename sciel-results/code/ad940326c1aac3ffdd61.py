"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_MODULE_START = _time.time()
_BUDGET_LIMIT = 120.0
_SAFETY_MARGIN = 2.0

def dispatch(ready_operations, time):
    # Deadline guard
    if _time.time() - _MODULE_START > _BUDGET_LIMIT - _SAFETY_MARGIN:
        return 0

    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        rw = op["remaining_work"]
        # Use minimum duration as a proxy for processing time urgency
        durations = list(op["machine_options"].values())
        op_dur = min(durations) if durations else 0.0
        
        if rw <= 0:
            cr = float('inf')
        else:
            # Critical Ratio: Slack / Remaining Work
            # Slack = Remaining Work - Current Time
            cr = (rw - time) / rw
            
        # Combine CR with a small factor of operation duration to break ties
        # We want to minimize the score
        score = cr + 0.01 * op_dur
        scores.append(score)
        
    min_idx = int(np.argmin(scores))
    return min_idx
# EVOLVE-BLOCK-END
