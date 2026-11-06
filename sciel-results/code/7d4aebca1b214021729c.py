"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global deadline guard
_START_TIME = time_module.time()
_DEADLINE = _START_TIME + 110.0  # 110s limit to be safe within 120s

def dispatch(ready_operations, time):
    # Check deadline
    if time_module.time() > _DEADLINE:
        return 0
    
    # Hybrid efficiency metric: Remaining Work / Current Min Duration
    # This favors jobs with short initial steps relative to their total remaining load,
    # reducing machine blocking.
    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Find the minimum duration among available machines
        if not machine_options:
            min_duration = 1.0 # Fallback
        else:
            min_duration = min(machine_options.values())
            if min_duration <= 0:
                min_duration = 1e-9
        
        # Score: High remaining work relative to short duration is good
        # We want to maximize this ratio to prioritize "efficient" progress
        score = remaining_work / min_duration
        scores.append(score)
        
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
