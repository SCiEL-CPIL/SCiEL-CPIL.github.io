"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Strategy: Model as multi-criteria ranking by normalizing each constraint's weight
    independently to [0,1] and computing the heuristic as the product of normalized prize
    and the inverse of the maximum normalized weight across all constraints.
    
    This prioritizes items that are highly valuable relative to their worst-case resource consumption.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # prizes and weights should be of shape (5, 100)
    n_constraints, n_items = prizes.shape
    
    # Normalize each constraint's weights to [0, 1]
    # For each constraint, normalize weights so min=0, max=1
    norm_weights = np.zeros_like(weights)
    for c in range(n_constraints):
        w_min = weights[c].min()
        w_max = weights[c].max()
        if w_max > w_min:
            norm_weights[c] = (weights[c] - w_min) / (w_max - w_min)
        else:
            norm_weights[c] = 0.0  # All weights equal for this constraint
    
    # Normalize prizes to [0, 1]
    p_min = prizes.min()
    p_max = prizes.max()
    if p_max > p_min:
        norm_prizes = (prizes - p_min) / (p_max - p_min)
    else:
        norm_prizes = np.ones_like(prizes) * 0.5
    
    # For each item, find the maximum normalized weight across all constraints
    # This represents the "worst-case" resource consumption
    max_norm_weight = norm_weights.max(axis=0)  # shape: (n_items,)
    
    # Compute normalized prize for each item (mean across constraints or use max)
    # Use the maximum prize across constraints as a measure of item value
    item_prize = norm_prizes.max(axis=0)  # shape: (n_items,)
    
    # Heuristic: product of normalized prize and inverse of (1 + max normalized weight)
    # Using 1 + max_norm_weight to avoid division by zero and ensure positivity
    heuristic = item_prize / (1.0 + max_norm_weight)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
