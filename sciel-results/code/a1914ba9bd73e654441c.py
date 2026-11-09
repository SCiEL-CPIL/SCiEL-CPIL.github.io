"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def _soft_min(x, tau=0.15):
    """Smooth approximation of min(x) via log-sum-exp."""
    if len(x) == 0:
        return 0.0
    neg_x = -x / tau
    max_neg_x = np.max(neg_x)
    if max_neg_x > 700:
        return float(np.min(x))
    log_sum = np.log(np.sum(np.exp(neg_x - max_neg_x))) + max_neg_x
    return -log_sum * tau


def dispatch(ready_operations, time):
    # Check deadline guard
    if _time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        min_durations = [min(op["machine_options"].values()) for op in ready_operations]
        return int(np.argmin(min_durations))

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Extract features
    rws = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    rem_ops = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=np.float64)

    # Use soft-min for a smooth surrogate of the minimum duration across machine options
    soft_min_durs = np.array([
        _soft_min(np.array(list(op["machine_options"].values()), dtype=np.float64), tau=0.15)
        if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # SPT term: 1 / duration favors short operations (local machine efficiency)
    eps = 1e-6
    spt_term = 1.0 / (soft_min_durs + eps)

    # Normalize each feature to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr)

    rw_norm = normalize(rws)
    spt_norm = normalize(spt_term)
    ops_norm = normalize(rem_ops)
    dur_norm = normalize(soft_min_durs)  # higher duration = worse, so we subtract

    # Weights: SPT is dominant (local efficiency), then remaining work (global urgency),
    # then remaining ops (closeness to completion), minus a small duration penalty
    w_spt = 1.2
    w_rw = 0.7
    w_ops = 0.25
    w_dur = 0.25

    scores = w_spt * spt_norm + w_rw * rw_norm + w_ops * ops_norm - w_dur * dur_norm

    # Deterministic tie-breaker
    scores += np.arange(n) * 1e-10

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
