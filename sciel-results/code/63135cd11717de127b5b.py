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

    # Geometric mean of ratios
    log_gm = log_ratios.mean(axis=0)

    # Total prize per item, normalized
    total_prize = prizes.sum(axis=0)
    max_prize = total_prize.max()
    norm_prize = total_prize / (max_prize + 1e-9)

    # Softmin and GM on ratio scale
    softmin_ratio = np.exp(softmin_log)
    gm_ratio = np.exp(log_gm)

    # Bottleneck slack: minimum remaining capacity ratio across all constraints
    capacity = 25.0
    slack = (capacity - weights) / weights  # Shape: (5, 100)
    bottleneck_slack = slack.min(axis=0)  # Shape: (100,)
    min_slack = bottleneck_slack.min()
    max_slack = bottleneck_slack.max()
    if max_slack > min_slack:
        norm_slack = (bottleneck_slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.ones_like(bottleneck_slack)

    # Greedy-set membership: simulate a greedy packing to identify which items
    # are likely to be selected early and thus "block" others. Items that are
    # NOT selected by a simple greedy heuristic may be less critical, while
    # items that ARE selected represent high-value choices that should be
    # prioritized. We compute a greedy score and use membership as a signal.
    
    # Simple greedy: repeatedly pick the item with best prize/weight sum ratio
    # that still fits, until no more fit. Mark selected items.
    n_items = weights.shape[1]
    selected = np.zeros(n_items, dtype=bool)
    remaining_cap = np.full(5, capacity, dtype=np.float64)
    
    # Work on a copy of ratios for greedy selection
    # Use total efficiency as greedy criterion
    greedy_score = ratios.sum(axis=0)
    
    for _ in range(n_items):
        if time.time() - t0 > _DEADLINE - 10.0:
            break
        # Find best unselected item that still fits
        best_idx = -1
        best_val = -1
        for i in range(n_items):
            if selected[i]:
                continue
            # Check if item fits
            if np.all(weights[:, i] <= remaining_cap + 1e-9):
                val = greedy_score[i]
                if val > best_val:
                    best_val = val
                    best_idx = i
        if best_idx == -1:
            break
        selected[best_idx] = True
        remaining_cap -= weights[:, best_idx]

    # Use greedy selection as a boost: selected items get a multiplier
    # This explicitly models blocking interactions
    greedy_boost = np.where(selected, 1.5, 0.8)

    # Final heuristic: combine efficiency, prize, slack, and greedy membership
    heuristic = (
        softmin_ratio
        * gm_ratio
        * (0.5 + 0.5 * norm_prize)
        * (0.5 + 0.5 * norm_slack)
        * greedy_boost
    )

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() - t0 > _DEADLINE - 5.0:
        # Return a simple fallback
        return prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
