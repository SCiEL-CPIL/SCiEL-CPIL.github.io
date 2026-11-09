"""Dispatch rule using adaptive rank-based priority with context-aware weighting."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE


def _rank(arr):
    """Compute ranks (0 to n-1) where 0 is smallest, n-1 is largest."""
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(len(arr))
    return ranks


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Quick deadline guard
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each op, compute min duration
    min_durations = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            min_durations[i] = 1.0
        else:
            best_m = min(mo.items(), key=lambda x: x[1])
            min_durations[i] = best_m[1]

    # Compute ranks for each feature (0 = lowest, n-1 = highest)
    rank_work = _rank(remaining_work)       # higher = more remaining work
    rank_ops = _rank(remaining_ops)          # higher = more ops left
    rank_spt = _rank(min_durations)          # higher = longer duration (we want lower)
    rank_spt_inv = (n - 1) - rank_spt        # invert: higher = shorter duration (better SPT)

    # Context-aware weighting:
    # - When few ops are ready (n <= 3), use more SPT to free machines quickly
    # - When many ops are ready (n >= 6), use more critical path to avoid bottlenecks
    # - Middle range: balanced approach
    if n <= 3:
        w_work = 0.40
        w_ops = 0.15
        w_spt = 0.45
    elif n <= 5:
        w_work = 0.50
        w_ops = 0.20
        w_spt = 0.30
    else:
        w_work = 0.60
        w_ops = 0.15
        w_spt = 0.25

    # Rank-based score: combine ranked features with context-aware weights
    score = w_work * rank_work + w_ops * rank_ops + w_spt * rank_spt_inv

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
