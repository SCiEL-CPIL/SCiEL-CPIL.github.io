"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each op, compute a smooth surrogate for the minimum duration (soft-min via log-sum-exp)
    # This captures "best machine" duration more smoothly than hard min
    min_durations = np.zeros(n, dtype=np.float64)
    best_machines = [None] * n

    for i, op in enumerate(ready_operations):
        if op["machine_options"]:
            durations = np.array(list(op["machine_options"].values()), dtype=np.float64)
            # Soft-min: -log-sum-exp(-durations / tau) * tau, with small tau for sharpness
            tau = 0.1
            neg_d = -durations / tau
            max_neg_d = np.max(neg_d)
            lse = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
            soft_min = -tau * lse
            min_durations[i] = soft_min
            # Also track the actual best machine for return
            best_machines[i] = min(op["machine_options"], key=op["machine_options"].get)
        else:
            min_durations[i] = 0.0
            best_machines[i] = 0

    # Number of machine options available
    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Normalize features to [0, 1] range for stable weighting
    def _norm(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    nw = _norm(remaining_works)
    no = _norm(remaining_ops)
    # Invert min_duration: lower duration is better
    md_inv = 1.0 - _norm(min_durations)
    nm = _norm(num_machines_available)

    # Composite score: emphasize remaining work strongly, with auxiliary signals
    # Weights tuned to balance LRPT dominance with local machine fit
    score = 2.0 * nw + 0.5 * no + 0.2 * md_inv + 0.1 * nm

    # Pick the argmax
    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
