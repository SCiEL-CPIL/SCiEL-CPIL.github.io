# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = None


def dispatch(ready_operations, time):
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()
    
    # Deadline guard: if we've been running too long, fall back to simple rule
    if _time.time() - _dispatch_start_time > 100:
        if not ready_operations:
            return 0
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    if not ready_operations:
        return 0
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        return (0, best_machine)
    
    # Compute a composite score for each ready operation
    scores = []
    best_machines = []
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        mo = op["machine_options"]
        
        # Find the best (fastest) machine for this operation
        min_dur = min(mo.values())
        best_machine = min(mo.items(), key=lambda x: x[1])[0]
        best_machines.append(best_machine)
        
        # Composite score: emphasize remaining work,
        # small penalty for long processing times (prefer quick completions)
        # Small bonus for jobs closer to completion
        score = rw * 1.0 - 0.05 * min_dur + 0.1 * (1.0 / max(ro, 1))
        scores.append(score)
    
    # Pick the operation with highest score
    idx = int(np.argmax(scores))
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
