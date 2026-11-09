"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_HARD_LIMIT = 110.0
_SOFT_MARGIN = 5.0

def _check_deadline():
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def _normalize(arr):
    """Min-max normalize to [0,1]. If all same, return zeros."""
    a = np.asarray(arr, dtype=np.float64)
    mn = a.min()
    mx = a.max()
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Extract features
    rem_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    rem_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    min_durs = np.zeros(n, dtype=np.float64)
    max_durs = np.zeros(n, dtype=np.float64)
    avg_durs = np.zeros(n, dtype=np.float64)
    num_machines = np.ones(n, dtype=np.int32)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durations = list(mo.values())
            min_durs[i] = min(durations)
            max_durs[i] = max(durations)
            avg_durs[i] = sum(durations) / len(durations)
            num_machines[i] = len(durations)
    
    # Heuristic 1: Most remaining work
    h1 = rem_works.copy()
    
    # Heuristic 2: Most remaining operations (urgency)
    h2 = rem_ops.copy()
    
    # Heuristic 3: Shorter min processing time (finish sooner)
    h3 = 1.0 / (min_durs + 1.0)
    
    # Heuristic 4: Fewer machine options (more constrained)
    h4 = 1.0 / (num_machines.astype(np.float64) + 1.0)
    
    # Heuristic 5: Work-to-duration ratio (long jobs on fast machines)
    h5 = rem_works / (avg_durs + 1.0)
    
    # Heuristic 6: Critical path estimate - remaining work weighted by remaining ops
    h6 = rem_works * (rem_ops + 1.0)
    
    # Heuristic 7: Variance of durations (lower variance = more predictable)
    dur_vars = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durations = np.array(list(mo.values()), dtype=np.float64)
            dur_vars[i] = durations.var()
    h7 = 1.0 / (dur_vars + 1.0)
    
    # Heuristic 8: Max duration penalty (avoid operations with very long max duration)
    h8 = 1.0 / (max_durs + 1.0)
    
    # Normalize all heuristics to [0,1]
    h1n = _normalize(h1)
    h2n = _normalize(h2)
    h3n = _normalize(h3)
    h4n = _normalize(h4)
    h5n = _normalize(h5)
    h6n = _normalize(h6)
    h7n = _normalize(h7)
    h8n = _normalize(h8)
    
    # Weighted combination tuned for open-shop makespan
    # High remaining work and operations are most important
    # Constraint tightness and predictability help avoid machine conflicts
    weights = np.array([1.2, 0.8, 0.5, 0.4, 0.7, 1.0, 0.3, 0.3], dtype=np.float64)
    
    scores = (
        weights[0] * h1n +
        weights[1] * h2n +
        weights[2] * h3n +
        weights[3] * h4n +
        weights[4] * h5n +
        weights[5] * h6n +
        weights[6] * h7n +
        weights[7] * h8n
    )
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
