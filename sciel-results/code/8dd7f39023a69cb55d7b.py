"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute heuristic attractiveness scores for items in a multi-constraint knapsack problem.
    
    Uses a multi-objective scoring approach:
    1. Prize efficiency: total prize / mean weight (average cost efficiency)
    2. Bottleneck resistance: penalty for items that are very heavy in any single constraint
    3. Constraint balance bonus: reward items that are uniformly light across constraints
    
    Args:
        prizes: array of shape (5, 100) with uniform values in [0, 1]
        weights: array of shape (5, 100) with uniform values in [0, 1]
    
    Returns:
        Array of shape (100,) with one attractiveness score per item
    """
    total_prize = prizes.sum(axis=0)  # shape: (100,)
    
    # Mean weight: average burden across constraints
    mean_w = weights.mean(axis=0)  # shape: (100,)
    
    # Max weight: the bottleneck constraint
    max_w = weights.max(axis=0)  # shape: (100,)
    
    # Variance of weights: low variance means the item is "balanced" across constraints
    # High variance means it's heavy in one constraint and light in others
    w_var = weights.var(axis=0)  # shape: (100,)
    
    # Component 1: Prize per unit mean weight (primary efficiency signal)
    eff_score = total_prize / (mean_w + 1e-9)
    
    # Component 2: Bottleneck penalty
    # Items with very high max_w relative to mean_w are risky
    # Use a soft penalty: exp(-alpha * (max_w - mean_w))
    bottleneck_penalty = np.exp(-1.5 * (max_w - mean_w))
    
    # Component 3: Balance bonus
    # Items with low variance across constraints are more predictable and easier to pack
    # Normalize variance to [0, ~0.08] range for uniform[0,1]
    balance_bonus = 1.0 / (1.0 + 5.0 * w_var)
    
    # Combine: primary efficiency scaled by bottleneck resistance and balance
    heuristic = eff_score * bottleneck_penalty * balance_bonus
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
