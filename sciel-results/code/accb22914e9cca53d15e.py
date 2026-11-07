"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute heuristic attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) with uniform values in [0, 1]
        weights: array of shape (5, 100) with uniform values in [0, 1]
    
    Returns:
        Array of shape (100,) with one attractiveness score per item
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # shape: (100,)
    
    # For multi-constraint knapsack, the bottleneck is the constraint that limits us most.
    # An item is constrained by its maximum weight across constraints (in relative terms).
    # We want to reward items with high prize relative to their worst-case weight burden.
    
    # Approach: Use a smooth max (log-sum-exp) of weights as a surrogate for the bottleneck constraint
    # This gives a smooth approximation of max(weights[item]) across constraints
    
    # Compute smooth max of weights across constraints for each item
    # Using log-sum-exp for numerical stability
    max_w = weights.max(axis=0)  # shape: (100,)
    shifted = weights - max_w[np.newaxis, :]  # shape: (5, 100)
    log_sum_exp_vals = max_w + np.log(np.exp(shifted).sum(axis=0) + 1e-15)  # shape: (100,)
    
    # Heuristic: prize per unit of bottleneck weight
    # Items with high total prize and low bottleneck weight are most attractive
    heuristic = total_prize / (log_sum_exp_vals + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
