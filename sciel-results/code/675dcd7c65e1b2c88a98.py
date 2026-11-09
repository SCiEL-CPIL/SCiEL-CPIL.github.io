import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill

def _soft_min(durations, temperature=0.3):
    """Soft-min via log-sum-exp: -T * log(sum(exp(-d/T)))
    Approaches hard min as T -> 0, but captures full distribution."""
    if len(durations) == 0:
        return 0.0
    if len(durations) == 1:
        return durations[0]
    arr = np.array(durations, dtype=np.float64)
    min_val = np.min(arr)
    # Numerically stable: shift by min
    shifted = arr - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = min(op["machine_options"].items(), key=lambda x: x[1])
        return (0, best_m)
    
    # Deadline guard: if we're running out of time, fall back to simple rule
    if _time.time() > _DEADLINE:
        # Fallback: pick op with max remaining work, min duration machine
        best_i = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] > best_rw:
                best_rw = op["remaining_work"]
                best_i = i
        best_op = ready_operations[best_i]
        if best_op["machine_options"]:
            bm = min(best_op["machine_options"].items(), key=lambda x: x[1])[0]
        else:
            bm = None
        return (best_i, bm)
    
    scores = np.zeros(n)
    best_machines = [None] * n
    
    # Precompute normalization constants
    max_rem_work = max(op["remaining_work"] for op in ready_operations)
    if max_rem_work == 0:
        max_rem_work = 1.0
    max_rem_ops = max(op["remaining_ops"] for op in ready_operations)
    if max_rem_ops == 0:
        max_rem_ops = 1.0
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            scores[i] = rem_work
            best_machines[i] = None
            continue
            
        # Extract durations and find best (min) machine
        durations = list(machine_options.values())
        min_dur = float('inf')
        best_m = None
        n_options = 0
        for m, d in machine_options.items():
            n_options += 1
            if d < min_dur:
                min_dur = d
                best_m = m
        
        # Soft-min: smooth surrogate that captures distribution
        # Lower temperature = closer to hard min; higher = more average-like
        sm_dur = _soft_min(durations, temperature=0.3)
        
        # Flexibility: fewer options = more constrained = higher priority
        flexibility = 1.0 / n_options if n_options > 0 else 0.0
        
        # Normalized remaining work (0 to 1)
        norm_work = rem_work / max_rem_work
        norm_ops = rem_ops / max_rem_ops
        
        # Heuristic Score using soft-min instead of hard min:
        # - High remaining work: critical path length
        # - Low soft-min duration: quick completion frees up resources (smooth)
        # - Fewer machine options: more constrained, schedule earlier
        # - More remaining ops: more work left, prioritize
        score = 2.5 * norm_work + 1.0 * flexibility - 0.5 * (sm_dur / max_rem_work) + 0.3 * norm_ops
        
        scores[i] = score
        best_machines[i] = best_m
        
    best_idx = int(np.argmax(scores))
    return (best_idx, best_machines[best_idx])
