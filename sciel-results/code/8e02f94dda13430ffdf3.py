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

    capacity = 25.0
    n_constraints, n_items = weights.shape

    # Per-constraint efficiency: prize/weight for each constraint
    # Shape: (5, 100)
    ratios = prizes / weights
    log_ratios = np.log(ratios)

    # Harmonic Mean of ratios (strong penalty for low ratio in any constraint)
    # H = 1 / (1/n * sum(1/r_i)) = n / sum(1/r_i)
    # In log space: log(H) = log(n) - log(sum(exp(-log(r_i))))
    # Using log-sum-exp for stability
    neg_log_ratios = -log_ratios
    max_neg = neg_log_ratios.max(axis=0)
    log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_log_ratios - max_neg), axis=0))
    log_harmonic_mean = np.log(n_constraints) - log_sum_exp_neg
    harmonic_mean = np.exp(log_harmonic_mean)

    # Geometric Mean of ratios (softer penalty)
    log_gm = log_ratios.mean(axis=0)
    gm_ratio = np.exp(log_gm)

    # Combine efficiency: Harmonic is stronger, so weight it more
    # If one ratio is very low, Harmonic drops significantly.
    efficiency = harmonic_mean * (0.5 + 0.5 * gm_ratio)

    # Total prize per item, normalized
    total_prize = prizes.sum(axis=0)
    max_prize = total_prize.max()
    if max_prize > 0:
        norm_prize = total_prize / max_prize
    else:
        norm_prize = np.ones(n_items)

    # Bottleneck slack: minimum remaining capacity ratio across all constraints
    # (25 - w_ic) / w_ic
    slack = (capacity - weights) / weights
    bottleneck_slack = slack.min(axis=0)
    min_slack = bottleneck_slack.min()
    max_slack = bottleneck_slack.max()
    if max_slack > min_slack:
        norm_slack = (bottleneck_slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.ones(n_items)

    # Per-constraint "how much of capacity this item uses"
    frac_used = weights / capacity
    max_frac = frac_used.max(axis=0)
    min_frac = max_frac.min()
    max_frac_max = max_frac.max()
    if max_frac_max > min_frac:
        norm_frac_inv = 1.0 - (max_frac - min_frac) / (max_frac_max - min_frac)
    else:
        norm_frac_inv = np.ones(n_items)

    # Feasibility component: prioritize items that fit easily
    feasibility = (0.3 + 0.7 * norm_slack) * (0.3 + 0.7 * norm_frac_inv)

    # Prize component
    prize_component = 0.5 + 0.5 * norm_prize

    # Greedy Boost: Identify items selected by a simple greedy heuristic
    # This helps focus the ACO on items that are part of a good "greedy core"
    selected = np.zeros(n_items, dtype=bool)
    remaining_cap = np.full(n_constraints, capacity, dtype=np.float64)
    
    # Greedy criterion: sum of per-constraint ratios (total efficiency)
    greedy_score = ratios.sum(axis=0)
    
    # Simple greedy selection
    # O(N^2) worst case, but N=100 is small
    for _ in range(n_items):
        if time.time() - t0 > _DEADLINE - 10.0:
            break
        best_idx = -1
        best_val = -1
        for i in range(n_items):
            if selected[i]:
                continue
            # Check if item fits in all constraints
            if np.all(weights[:, i] <= remaining_cap + 1e-9):
                val = greedy_score[i]
                if val > best_val:
                    best_val = val
                    best_idx = i
        if best_idx == -1:
            break
        selected[best_idx] = True
        remaining_cap -= weights[:, best_idx]

    # Boost items selected by greedy, penalize others slightly
    # Stronger boost to guide ants towards the greedy solution's items
    greedy_boost = np.where(selected, 1.8, 0.8)

    # Final heuristic
    heuristic = efficiency * feasibility * prize_component * greedy_boost

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() - t0 > _DEADLINE - 5.0:
        return prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
