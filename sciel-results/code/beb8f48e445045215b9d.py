"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s kill

def _check_deadline():
    return _time.time() - _start_time < _DEADLINE_SECONDS

def _soft_min(values, temperature=0.5):
    """Soft-min via log-sum-exp: smooth surrogate for min()."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    m = arr.min()
    if m == 0:
        return 0.0
    # Soft min = -T * log(sum(exp(-x/T)))
    exp_vals = np.exp(-(arr - m) / temperature)
    s = exp_vals.sum()
    if s == 0:
        return m
    return m - temperature * np.log(s)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to timeout, return first ready op
    if not _check_deadline():
        return 0

    n = len(ready_operations)
    scores = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op.get("remaining_ops", 0)
        durations = list(op["machine_options"].values())

        if not durations:
            d_min = 0.0
            d_mean = 0.0
        else:
            d_min = _soft_min(durations, temperature=0.3)
            d_mean = float(np.mean(durations))

        # Composite score:
        # - remaining_work: LPT component (larger jobs first)
        # - remaining_ops: critical path length proxy (more ops = more urgency)
        # - soft_min duration: immediate processing time (larger = more blocking)
        # - time-aware: operations with less remaining work relative to time are "later"
        #   but we want to start big jobs early, so keep LPT dominant
        score = (1.0 * rw
                 + 0.8 * ro * d_min       # critical path: ops remaining * avg duration
                 + 0.4 * d_min            # immediate LPT
                 + 0.1 * d_mean)          # flexibility penalty (mean > min means constrained)

        scores[i] = score

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
