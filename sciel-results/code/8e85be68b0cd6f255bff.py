"""Dispatch rule using adaptive rank-based priority with bottleneck-aware slack maximization."""
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


def _soft_min(arr, temperature=1.0):
    """
    Smooth surrogate for min using log-sum-exp with negative sign.
    soft_min(x) = -log(sum(exp(-x_i / T))) * T
    As T -> 0, this approaches the true min.
    """
    neg_arr = -arr / temperature
    max_neg = np.max(neg_arr)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_arr - max_neg)))
    return -log_sum * temperature


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

    # For each op, compute min duration and soft-min duration
    min_durations = np.empty(n, dtype=np.float64)
    all_durations = []
    machine_flexibility = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            min_durations[i] = 1.0
            all_durations.append([1.0])
            machine_flexibility[i] = 0.0
        else:
            durations = list(mo.values())
            min_durations[i] = min(durations)
            all_durations.append(durations)
            # Flexibility: number of machine options (more options = more flexible)
            machine_flexibility[i] = len(mo)

    # Compute soft-min for each operation (smooth surrogate for min duration)
    soft_min_durations = np.empty(n, dtype=np.float64)
    for i, durs in enumerate(all_durations):
        if len(durs) == 1:
            soft_min_durations[i] = durs[0]
        else:
            soft_min_durations[i] = _soft_min(np.array(durs, dtype=np.float64), temperature=0.5)

    # Critical path feature: remaining work per remaining operation
    critical_path = np.where(remaining_ops > 0, remaining_work / np.maximum(remaining_ops, 1), remaining_work)

    # WSPT ratio: remaining work per unit min processing time (urgency measure)
    wspt_ratio = remaining_work / (min_durations + 1e-9)

    # Compute ranks for each feature (0 = lowest, n-1 = highest)
    rank_work = _rank(remaining_work)       # higher = more remaining work
    rank_ops = _rank(remaining_ops)          # higher = more ops left
    rank_cp = _rank(critical_path)           # higher = more critical
    rank_wspt = _rank(wspt_ratio)            # higher = more urgent (more work per time)
    rank_spt = _rank(soft_min_durations)     # higher = longer duration
    rank_spt_inv = (n - 1) - rank_spt        # invert: higher = shorter duration (better SPT)
    rank_flex = _rank(machine_flexibility)   # higher = more flexible (more machine options)
    rank_flex_inv = (n - 1) - rank_flex      # invert: higher = less flexible (more constrained)

    # Bottleneck slack estimation:
    # For each operation, estimate the "slack" if we start it now:
    # slack_i = remaining_work_i - (time * 1.0)  -- simplified: we don't have due dates
    # Instead, use a proxy: operations with high remaining work AND low flexibility
    # are more likely to be on the critical path and cause bottlenecks.
    # We want to maximize the minimum slack, so we prioritize operations that
    # have high urgency (WSPT) but are constrained (low flexibility).
    
    # Slack proxy: high WSPT + low flexibility = high bottleneck risk
    # We want to pick the op that, when started, most reduces bottleneck risk.
    # This is approximated by: high WSPT rank * low flexibility rank (multiplicative interaction)
    bottleneck_risk = rank_wspt * rank_flex_inv  # high = high risk, should prioritize

    # Context-aware weighting:
    # - When few ops are ready (n <= 3), use more SPT to free machines quickly
    # - When many ops are ready (n >= 6), use more critical path to avoid bottlenecks
    # - Middle range: balanced approach
    if n <= 3:
        w_work = 0.25
        w_ops = 0.10
        w_cp = 0.10
        w_wspt = 0.25
        w_spt = 0.30
        w_bottleneck = 0.00
    elif n <= 5:
        w_work = 0.20
        w_ops = 0.10
        w_cp = 0.15
        w_wspt = 0.25
        w_spt = 0.20
        w_bottleneck = 0.10
    else:
        w_work = 0.15
        w_ops = 0.10
        w_cp = 0.20
        w_wspt = 0.25
        w_spt = 0.15
        w_bottleneck = 0.15

    # Normalize bottleneck_risk to [0, 1] for consistent scaling
    max_bottleneck = np.max(bottleneck_risk) if np.max(bottleneck_risk) > 0 else 1.0
    bottleneck_norm = bottleneck_risk / max_bottleneck

    # Rank-based score: combine ranked features with context-aware weights
    # Use multiplicative interaction between WSPT and bottleneck for stronger signal
    score = (
        w_work * rank_work
        + w_ops * rank_ops
        + w_cp * rank_cp
        + w_wspt * rank_wspt
        + w_spt * rank_spt_inv
        + w_bottleneck * bottleneck_norm * (n - 1)  # scale to rank range
    )

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
