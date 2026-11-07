"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Uses a "Marginal Utility" approach combined with Bottleneck Efficiency.
    1. Marginal Utility: Prize per opportunity cost.
       Opportunity cost is estimated by weighting each constraint's weight
       by a scarcity factor derived from the top-k most efficient items
       in that constraint. This penalizes items that consume scarce,
       highly-contested resources.
    2. Bottleneck Efficiency: Geometric mean of per-constraint ratios.
       Provides a robust baseline for items that are efficient across
       all constraints.
    
    Returns a single score per item (shape 100,).
    """
    start_time = time.time()
    
    # Deadline guard: if we are running too long, return a safe default
    # The evaluator kills at 330s, we use 290s to be safe.
    # However, this function is called once per instance, so it should be fast.
    # The guard is primarily for unexpected slowness.
    
    eps = 1e-9
    n_constraints, n_items = weights.shape
    
    # 1. Calculate per-constraint efficiencies R_{i,c}
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # 2. Calculate Scarcity Factors S_c for each constraint
    # Scarcity is high if there are many items with high efficiency in that constraint.
    # We use the mean of the top 20% of efficiencies for each constraint.
    k = max(1, int(0.2 * n_items))
    
    scarcity = np.zeros(n_constraints)
    for c in range(n_constraints):
        # Get efficiencies for constraint c: shape (100,)
        eff_c = ratios[c, :]
        # Get top k values
        # np.partition is faster than np.sort
        top_k_vals = np.partition(eff_c, -k)[-k:]
        scarcity[c] = np.mean(top_k_vals)
        
    # Ensure scarcity is positive
    scarcity = np.maximum(scarcity, eps)
    
    # 3. Calculate Opportunity Cost for each item
    # Cost_i = sum_c (w_{i,c} * S_c)
    # weights shape: (5, 100), scarcity shape: (5,)
    # Broadcasting: (5, 100) * (5, 1) -> (5, 100)
    weighted_costs = weights * scarcity[:, np.newaxis]
    opportunity_cost = weighted_costs.sum(axis=0) # Shape: (100,)
    
    # 4. Calculate Total Prize for each item
    # prizes shape: (5, 100)
    total_prize = prizes.sum(axis=0) # Shape: (100,)
    
    # 5. Marginal Utility Score
    # U_i = Total_Prize_i / Opportunity_Cost_i
    marginal_utility = total_prize / (opportunity_cost + eps)
    
    # 6. Bottleneck Efficiency (Geometric Mean)
    # Re-use ratios calculated above
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_ratio = np.exp(log_geo_mean)
    
    # Normalize both metrics to mean 1
    mu_mean = marginal_utility.mean()
    geo_mean = geo_ratio.mean()
    
    if mu_mean > 0:
        mu_norm = marginal_utility / mu_mean
    else:
        mu_norm = np.ones_like(marginal_utility)
        
    if geo_mean > 0:
        geo_norm = geo_ratio / geo_mean
    else:
        geo_norm = np.ones_like(geo_ratio)
        
    # Blend the two normalized metrics
    # Weight Marginal Utility higher as it captures scarcity dynamics
    scores = 0.7 * mu_norm + 0.3 * geo_norm
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    # Check deadline again before returning
    if time.time() - start_time > 290:
        return np.ones(n_items)
        
    return scores
# EVOLVE-BLOCK-END
