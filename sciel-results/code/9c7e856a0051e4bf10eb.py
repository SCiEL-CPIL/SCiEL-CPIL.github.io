"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    
    scores = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        rem_work = float(op["remaining_work"])
        rem_ops = max(int(op["remaining_ops"]), 1)
        
        # Minimum duration across machine options
        machine_opts = op["machine_options"]
        if machine_opts:
            min_dur = min(machine_opts.values())
        else:
            min_dur = 0.0
        
        # Composite score:
        # 1. Remaining work on job (urgency - jobs with more work left are more critical)
        # 2. Operation duration (longer ops keep machines busy, reduce idle time)
        # 3. Average remaining work per op (balance - avoid over-focusing on one job)
        avg_rem = rem_work / rem_ops
        
        score = 1.0 * rem_work + 0.8 * min_dur + 0.5 * avg_rem
        scores[i] = score
    
    # Add small tie-breaking: prefer operations with fewer remaining ops (closer to completion)
    # This helps finish jobs and free up resources
    for i, op in enumerate(ready_operations):
        rem_ops = max(int(op["remaining_ops"]), 1)
        scores[i] -= 0.05 * rem_ops
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
