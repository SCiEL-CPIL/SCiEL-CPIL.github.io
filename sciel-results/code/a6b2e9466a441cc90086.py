"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _deadline_exceeded():
    return _time.time() - _start_time > _DEADLINE

def _soft_min(durations, beta=10.0):
    """Smooth minimum via log-sum-exp: -log(sum(exp(-beta*x_i)))/beta"""
    d = np.asarray(durations, dtype=np.float64)
    if d.size == 0:
        return 1.0
    if d.size == 1:
        return float(d[0])
    # Numerically stable soft-min
    min_d = np.min(d)
    shifted = d - min_d
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_d + log_sum / beta
    return float(soft_min)

def _rank(scores):
    """Return rank of each score (1-based, higher score = higher rank)."""
    if len(scores) == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Check deadline early
    if _deadline_exceeded():
        return 0

    # Precompute scores for all operations
    soft_mins = np.empty(n, dtype=np.float64)
    remaining_works = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        if not durations:
            soft_mins[i] = 1.0
        else:
            soft_mins[i] = _soft_min(durations, beta=10.0)
        remaining_works[i] = float(op["remaining_work"])
        remaining_ops[i] = float(op["remaining_ops"])

    # Component 1: WSPT-like score (urgency / efficiency)
    wspt_scores = remaining_works / (soft_mins + 1e-9)

    # Component 2: Critical path length (remaining work)
    critical_scores = remaining_works

    # Component 3: Shortest processing time (favor short ops to reduce makespan)
    spt_scores = 1.0 / (soft_mins + 1e-9)

    # Component 4: Remaining ops (favor jobs with more work left - urgency)
    urgency_scores = remaining_ops

    # Context-aware weighting: adapt weights based on current time
    # Early: emphasize critical path and remaining ops
    # Late: emphasize short processing time to finish quickly
    t_norm = min(time / 1000.0, 1.0)  # Normalize time to [0,1]
    w_wspt = 0.3 * (1 - t_norm) + 0.2 * t_norm
    w_critical = 0.3 * (1 - t_norm) + 0.1 * t_norm
    w_spt = 0.2 * (1 - t_norm) + 0.4 * t_norm
    w_urgency = 0.2 * (1 - t_norm) + 0.3 * t_norm

    # Combine scores
    combined = (w_wspt * wspt_scores +
                w_critical * critical_scores +
                w_spt * spt_scores +
                w_urgency * urgency_scores)

    # Rank-based adjustment: use ranks to make the rule more robust to scale
    ranks = _rank(combined)
    # Blend raw score with rank (0.7 raw, 0.3 rank)
    final_scores = 0.7 * combined + 0.3 * ranks

    best_idx = int(np.argmax(final_scores))

    if _deadline_exceeded():
        return best_idx

    return best_idx
# EVOLVE-BLOCK-END
