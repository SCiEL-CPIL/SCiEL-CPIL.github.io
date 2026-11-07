"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    prizes: shape (5, 100), values in [0, 1]
    weights: shape (5, 100), values in [0, 1]
    Returns: array of shape (100,) with attractiveness scores
    """
    n_constraints, n_items = prizes.shape
    eps = 1e-9
    
    # Total prize per item
    item_prizes = prizes.sum(axis=0)
    
    # For each constraint, compute scarcity based on top-k efficiency
    k = 10  # top-k most efficient items per constraint
    scarcity = np.zeros(n_constraints)
    
    for j in range(n_constraints):
        # Efficiency for constraint j: prize / weight
        eff = prizes[j] / (weights[j] + eps)
        # Get top-k efficiencies
        top_k_eff = np.sort(eff)[-k:]
        # Scarcity: mean of top-k efficiencies (higher = more scarce/valuable)
        scarcity[j] = np.mean(top_k_eff)
    
    # Compute weighted cost for each item based on scarcity
    # cost[i] = sum_j(weights[i,j] * scarcity[j])
    weighted_cost = weights.T @ scarcity  # shape (n_items,)
    
    # Heuristic: prize / weighted_cost
    heuristic = item_prizes / (weighted_cost + eps)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
# EVOLVE-BLOCK-END
