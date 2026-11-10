"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Uses a combination of:
    1. Bottleneck slack: min_c (25 - w_ic) / w_ic - how much slack remains in the tightest constraint
    2. Total prize: sum of prizes across constraints
    3. Lagrangian-style dual pricing: penalizes items that consume scarce resources
    
    Returns an array of shape (100,) with one score per item.
    """
    t0 = time.time()
    
    # Ensure correct shapes and positive values
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected shapes (5, 100)")
    
    # Ensure positive values
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    capacity = 25.0
    n_constraints = 5
    n_items = 100
    
    # 1. Bottleneck slack score: for each item, the minimum remaining capacity ratio
    #    if it were added first: min_c (capacity - w_ic) / w_ic
    #    Higher means more feasible
    slack_ratios = (capacity - weights) / weights  # shape (5, 100)
    bottleneck_slack = np.min(slack_ratios, axis=0)  # shape (100,)
    bottleneck_slack = np.maximum(bottleneck_slack, 0.0)
    
    # 2. Total prize per item
    total_prize = np.sum(prizes, axis=0)  # shape (100,)
    
    # 3. Per-constraint efficiency: prize_c / weight_c for each constraint
    efficiency = prizes / weights  # shape (5, 100)
    
    # 4. Weighted efficiency: weight each constraint by its scarcity
    #    Scarcity: how much of the capacity is used on average across items
    avg_weight_per_constraint = np.mean(weights, axis=1)  # shape (5,)
    scarcity = avg_weight_per_constraint / capacity  # shape (5,) - higher means more scarce
    # Normalize scarcity to [0, 1]
    scarcity = scarcity / (scarcity.max() + 1e-9)
    
    # Weighted efficiency score: sum over constraints of scarcity_c * efficiency_c
    weighted_efficiency = np.sum(scarcity[:, np.newaxis] * efficiency, axis=0)  # shape (100,)
    
    # 5. Combine scores
    #    Primary: total_prize * bottleneck_slack (prioritize high-value items that fit easily)
    #    Secondary: weighted_efficiency (prioritize items that are efficient on scarce constraints)
    
    # Normalize components to comparable scales
    total_prize_norm = total_prize / (total_prize.max() + 1e-9)
    bottleneck_slack_norm = bottleneck_slack / (bottleneck_slack.max() + 1e-9)
    weighted_eff_norm = weighted_efficiency / (weighted_efficiency.max() + 1e-9)
    
    # Combine: emphasize bottleneck slack and total prize, with efficiency as tiebreaker
    score = (total_prize_norm * 0.5 + bottleneck_slack_norm * 0.3 + weighted_eff_norm * 0.2)
    
    # Ensure all values are positive
    score = np.maximum(score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
