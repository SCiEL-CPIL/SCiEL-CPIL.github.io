"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 295.0  # safety margin before 300s limit


def _softmin(x, beta=1.0):
    """Smooth minimum: softmin(x) = -logsumexp(-x/beta)*beta."""
    neg = -x / beta
    m = np.max(neg, axis=0)
    # Numerical stability
    m = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(neg - m), axis=0)
    s = np.where(s > 0, s, 1e-300)
    return -beta * (m + np.log(s))


def heuristics(prizes, weights):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if prizes.shape != weights.shape:
        raise ValueError("shape mismatch")

    n_constraints, n_items = prizes.shape

    # Sanitize: ensure positive, finite
    prizes = np.where(np.isfinite(prizes) & (prizes > 0), prizes, 1e-9)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1e-9)

    # Per-constraint efficiency ratio: prize_c / weight_c
    ratios = prizes / weights  # shape (C, N)

    # Signal 1: soft-min across constraints of the log-ratio.
    # Items with a very poor ratio on ANY constraint get penalized.
    log_ratios = np.log(ratios)
    # softmin of log_ratios across constraints
    softmin_log_ratio = _softmin(log_ratios, beta=0.5)  # shape (N,)

    # Signal 2: bottleneck-slack feasibility score.
    # For each item, find the constraint where it consumes the most relative capacity.
    # slack_c = (capacity - weight_ic) / capacity  for each constraint c
    # bottleneck_slack = min_c slack_c  (smoothed)
    capacity = 25.0
    slack = (capacity - weights) / capacity  # shape (C, N), values in (~0, 1)
    slack = np.clip(slack, 1e-9, 1.0)
    # softmin of slack across constraints
    softmin_slack = _softmin(slack, beta=0.3)  # shape (N,)

    # Combine: geometric mean (product in log space) of the two signals
    # Heuristic = exp(softmin_log_ratio) * softmin_slack
    # exp(softmin_log_ratio) is a smooth geometric-mean-like efficiency
    efficiency = np.exp(softmin_log_ratio)
    efficiency = np.where(np.isfinite(efficiency) & (efficiency > 0), efficiency, 1e-9)

    heuristic = efficiency * softmin_slack
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Safety: ensure positive and finite
    heuristic = np.clip(heuristic, 1e-9, 1e6)

    # Deadline guard (computation is fast, but be safe)
    if time.time() - t0 > _DEADLINE:
        return heuristic

    return heuristic
# EVOLVE-BLOCK-END
