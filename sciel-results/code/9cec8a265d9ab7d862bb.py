"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for 3 fixed instances with 100 items and 5 constraints.
    
    Uses a soft-min surrogate to capture the binding constraint:
    - For each item, compute the 'bottleneck' constraint via log-sum-exp on normalized weights
    - Weight prize by inverse of this bottleneck to favor items that don't stress the tightest constraint
    - Also incorporate a prize-per-total-weight ratio as a secondary signal
    """
    start_time = time.time()
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Normalize weights by capacity to get [0, 1/25] range
    # This makes the "slack" interpretation clearer
    w_norm = weights / capacity  # shape (5, 100)
    
    # Soft-min of weights across constraints for each item
    # We want to identify which items are "expensive" in at least one constraint
    # Use log-sum-exp on the weights to get a smooth max, then use that as a proxy for bottleneck
    # Actually, for knapsack, the bottleneck is the constraint that's most likely to be violated
    # An item is "bad" if it has high weight in ANY constraint
    # So we want to penalize items with high max weight across constraints
    
    # Use log-sum-exp for a smooth max (soft-max)
    # soft_max(w) = log(sum(exp(w_i)))
    # To avoid overflow, subtract the max first
    w_max = w_norm.max(axis=0)  # shape (100,)
    exp_w = np.exp(w_norm - w_max[np.newaxis, :])
    log_sum_exp = np.log(exp_w.sum(axis=0)) + w_max  # shape (100,)
    
    # The soft-max gives us a smooth approximation of the maximum weight across constraints
    # Higher values mean the item is expensive in at least one constraint
    # We want to penalize this, so use inverse
    
    # Also compute the mean weight as a secondary signal
    w_mean = w_norm.mean(axis=0)  # shape (100,)
    
    # Combine: use a weighted combination of soft-max and mean
    # The soft-max captures the bottleneck, the mean captures overall cost
    cost_soft = 0.7 * log_sum_exp + 0.3 * w_mean
    
    # Prize: use the mean prize across constraints as a scalar prize
    prize_mean = prizes.mean(axis=0)  # shape (100,)
    
    # Heuristic: prize / cost, with a floor to avoid division by zero
    # Add a small constant to cost to avoid division by zero
    heuristic = prize_mean / (cost_soft + 1e-6)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:  # Safety margin before 300s limit
        # Return a simple fallback if we're running out of time
        return prize_mean / (w_mean + 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
