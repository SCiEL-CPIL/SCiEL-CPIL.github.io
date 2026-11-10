"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0  # seconds wall-clock limit


def heuristics(prizes, weights):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Ensure positive values to avoid division issues
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)

    # Per-constraint efficiency: prize/weight for each constraint
    # Shape: (5, 100)
    ratios = prizes / weights

    # Log of ratios for smooth operations
    log_ratios = np.log(ratios)

    # Soft-min across constraints (smooth surrogate for min-ratio)
    # Using log-sum-exp trick: softmin(x) = -logsumexp(-x)
    # We want to penalize items that are poor on any single constraint
    neg_log_ratios = -log_ratios
    max_neg = neg_log_ratios.max(axis=0)
    log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_log_ratios - max_neg), axis=0))
    softmin_log = -log_sum_exp_neg  # This is log(geometric mean of ratios) = softmin of log ratios

    # Geometric mean of ratios (smooth version via log-sum-exp with beta=1)
    # GM = exp(mean(log(ratios)))
    log_gm = log_ratios.mean(axis=0)

    # Total prize per item, normalized
    total_prize = prizes.sum(axis=0)
    max_prize = total_prize.max()
    norm_prize = total_prize / (max_prize + 1e-9)

    # Combine: softmin (penalizes worst constraint) * GM * normalized prize
    # Use exp to get back to ratio scale for softmin component
    softmin_ratio = np.exp(softmin_log)
    gm_ratio = np.exp(log_gm)

    # Bottleneck slack: minimum remaining capacity ratio across all constraints
    # Capacity is 25 for each constraint
    capacity = 25.0
    # For each item, compute (capacity - weight) / weight for each constraint
    # This measures how much "room" remains after adding this item
    slack = (capacity - weights) / weights  # Shape: (5, 100)
    # The bottleneck slack is the minimum across constraints
    bottleneck_slack = slack.min(axis=0)  # Shape: (100,)
    # Normalize bottleneck slack to [0, 1] range for blending
    min_slack = bottleneck_slack.min()
    max_slack = bottleneck_slack.max()
    if max_slack > min_slack:
        norm_slack = (bottleneck_slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.ones_like(bottleneck_slack)

    # Final heuristic: blend of softmin efficiency, geometric mean efficiency, prize, and slack
    # Weight the slack component to prioritize items that fit easily into tight constraints
    heuristic = softmin_ratio * gm_ratio * (0.5 + 0.5 * norm_prize) * (0.5 + 0.5 * norm_slack)

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() - t0 > _DEADLINE - 5.0:
        # Return a simple fallback
        return prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
