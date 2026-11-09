"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def dispatch(ready_operations, time):
    # Check deadline guard
    if _time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick the operation with minimum processing time on any machine
        min_durations = [min(op["machine_options"].values()) for op in ready_operations]
        return int(np.argmin(min_durations))
    
    if not ready_operations:
        return 0
    
    # Combined heuristic: prioritize operations with high remaining work
    # but also consider machine load balancing and urgency
    scores = []
    for i, op in enumerate(ready_operations):
        # Remaining work component (higher is better)
        rw = op.get("remaining_work", 0)
        
        # Minimum processing time (lower is better - SPT-like)
        min_dur = min(op["machine_options"].values()) if op["machine_options"] else 0
        
        # Number of remaining operations (urgency)
        rem_ops = op.get("remaining_ops", 0)
        
        # Combine: weight remaining work heavily, but also consider short processing times
        # and urgency from remaining operations
        score = rw * 1.0 - min_dur * 0.5 + rem_ops * 0.1
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
