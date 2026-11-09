import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 115.0  # safety margin before 150s kill

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TS > DEADLINE_S:
        return 0
    
    # Strategy: Greedy Bottleneck Elimination (LPT)
    # Prioritize the operation with the MAXIMUM minimum processing time.
    # This aggressively schedules long, inflexible operations early to prevent
    # them from becoming critical path bottlenecks later.
    
    scores = []
    for op in ready_operations:
        # Get the minimum duration this operation can take
        # (the simulator will assign the best available machine)
        min_dur = min(op["machine_options"].values())
        # LPT: We want to pick the operation with the LARGEST min_dur
        # So we maximize this score (use negative for argmax, or just argmax).
        scores.append(min_dur)
        
    # Return index of the operation with the maximum score (longest min duration)
    return int(np.argmax(scores))
