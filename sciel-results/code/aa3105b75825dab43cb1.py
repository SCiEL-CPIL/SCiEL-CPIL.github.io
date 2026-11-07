import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each item under each constraint
        weights: array of shape (5, 100) - weight/cost for each item under each constraint
    
    Returns:
        array of shape (100,) - heuristic attractiveness scores (higher = more attractive)
    """
    n_constraints, n_items = prizes.shape
    
    # Total prize per item (sum across all constraints)
    total_prize = np.sum(prizes, axis=0)  # shape: (100,)
    
    # Total weight per constraint (to understand resource scarcity)
    total_weight_per_constraint = np.sum(weights, axis=1)  # shape: (5,)
    total_weight_per_constraint = np.where(total_weight_per_constraint > 0, 
                                            total_weight_per_constraint, 1.0)
    
    # Normalized weight: what fraction of each constraint's total does this item consume?
    # This helps identify items that are "expensive" relative to available capacity
    normalized_weights = weights / total_weight_per_constraint[:, np.newaxis]  # shape: (5, 100)
    
    # Multiple cost metrics:
    # 1. Sum of normalized weights (total resource consumption)
    sum_normalized_cost = np.sum(normalized_weights, axis=0)  # shape: (100,)
    
    # 2. Max normalized weight (most binding constraint for this item)
    max_normalized_cost = np.max(normalized_weights, axis=0)  # shape: (100,)
    
    # 3. Geometric mean of normalized weights (penalizes imbalance)
    #    Items with very different weights across constraints are less flexible
    eps = 1e-10
    log_weights = np.log(normalized_weights + eps)
    geo_mean_cost = np.exp(np.mean(log_weights, axis=0))  # shape: (100,)
    
    # 4. Variance of normalized weights (measures imbalance)
    variance_cost = np.var(normalized_weights, axis=0)  # shape: (100,)
    
    # Combine costs into a single effective cost:
    # - Use a weighted combination that emphasizes both total cost and imbalance
    # - Items with high variance are harder to pack efficiently
    effective_cost = (0.4 * sum_normalized_cost + 
                      0.3 * max_normalized_cost + 
                      0.3 * geo_mean_cost)
    
    # Add a penalty for imbalance: items with high variance get lower scores
    # Scale variance to be comparable to other costs
    variance_scale = np.mean(sum_normalized_cost) + 1e-10
    imbalance_penalty = variance_cost / variance_scale
    
    # Final cost: higher for items that are expensive or imbalanced
    final_cost = effective_cost * (1.0 + 2.0 * imbalance_penalty)
    
    # Heuristic: prize divided by effective cost
    # Add small epsilon to avoid division by zero
    heuristic = total_prize / (final_cost + 1e-9)
    
    # Apply power scaling to sharpen the heuristic
    # This makes high-value items significantly more attractive
    heuristic = heuristic ** 1.2
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
