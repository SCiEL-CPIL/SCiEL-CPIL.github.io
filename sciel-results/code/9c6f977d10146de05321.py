"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import math

# Deadline guard
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0

def _elapsed():
    return _time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        # Fast fallback: pick max remaining_work
        best_idx = 0
        best_rw = -1.0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        return int(best_idx)
    
    n = len(ready_operations)
    
    # We compute a priority score for each ready operation.
    # Key insight: model resource contention by penalizing operations
    # whose minimum duration is large relative to the workload, and
    # strongly reward constrained operations (few machine choices).
    # Also use a smooth (soft-min) surrogate for min duration.
    
    # Pre-extract features into arrays for vectorized computation
    remaining_works = np.zeros(n, dtype=float)
    min_durs = np.ones(n, dtype=float)
    soft_min_durs = np.ones(n, dtype=float)
    num_opts = np.ones(n, dtype=float)
    remaining_ops_arr = np.ones(n, dtype=float)
    valid = np.ones(n, dtype=bool)
    
    for i, op in enumerate(ready_operations):
        remaining_works[i] = op.get("remaining_work", 0.0)
        machine_options = op.get("machine_options", {})
        remaining_ops_arr[i] = max(op.get("remaining_ops", 0), 1)
        
        if not machine_options:
            valid[i] = False
            continue
        
        durations = np.array(list(machine_options.values()), dtype=float)
        num_opts[i] = len(durations)
        
        # Clamp to positive
        durations = np.clip(durations, 0.1, None)
        
        min_durs[i] = np.min(durations)
        
        # Soft-min via log-sum-exp
        min_d = min_durs[i]
        exp_terms = np.exp(-(durations - min_d))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_durs[i] = min_d - log_sum
        
        # Ensure positive
        if soft_min_durs[i] <= 0:
            soft_min_durs[i] = 0.1
    
    # Compute scores vectorized
    scores = np.full(n, float('-inf'), dtype=float)
    
    for i in range(n):
        if not valid[i]:
            continue
        
        rw = remaining_works[i]
        mn = min_durs[i]
        sm = soft_min_durs[i]
        no = num_opts[i]
        rop = remaining_ops_arr[i]
        
        # Feature 1: Constraint tightness (inverse of number of options)
        f1 = 1.0 / no
        
        # Feature 2: Remaining work urgency
        f2 = rw
        
        # Feature 3: Inverse soft-min duration (SPT-like)
        f3 = 1.0 / sm
        
        # Feature 4: Inverse remaining ops (critical path proxy)
        f4 = 1.0 / rop
        
        # Feature 5: WSPT proxy = work / duration
        f5 = rw / sm
        
        # Feature 6: Contention penalty - operations with longer min duration
        # create more load; penalize by normalizing against total work
        total_work = np.sum(remaining_works[valid])
        if total_work <= 0:
            total_work = 1.0
        f6 = mn / total_work  # small fraction = less congesting
        
        # Feature 7: Ratio of soft-min to hard-min (how many "good" machines)
        # If soft-min is close to hard-min, there's a clear best machine
        # If soft-min is much larger, many machines are similarly slow
        f7 = mn / sm if sm > 0 else 1.0
        
        # Weighted combination
        # Emphasize: WSPT, constraint tightness, SPT, and contention
        w1 = 2.0   # constraint tightness
        w2 = 1.0   # remaining work
        w3 = 2.5   # SPT (soft)
        w4 = 0.5   # remaining ops
        w5 = 4.0   # WSPT
        w6 = 1.5   # contention penalty (negate: prefer lower f6)
        w7 = 1.0   # machine clarity
        
        score = (w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 
                 + w5 * f5 - w6 * f6 + w7 * f7)
        
        scores[i] = score
    
    # Pick the operation with the highest score
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return 0
    
    best_idx = valid_indices[np.argmax(scores[valid_indices])]
    
    return int(best_idx)
# EVOLVE-BLOCK-END
