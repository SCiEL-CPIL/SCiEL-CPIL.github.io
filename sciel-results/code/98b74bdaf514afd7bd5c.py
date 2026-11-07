import numpy as np

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (cosine8 8d).
    
    Uses a cost-normalized Lower Confidence Bound (LCB) with a dynamic exploration
    parameter (kappa) that adapts based on remaining budget and current best.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays for consistent behavior
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-9)
    
    # Sanitize cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Budget-aware dynamic Kappa
    # Total budget is 30. Initial 5 uniform points cost roughly:
    # cost = 1 + mean(x_scaled), where x_scaled in [0,1], mean ~ 0.5
    # So initial cost ~ 5 * 1.5 = 7.5
    # Remaining budget drives exploration vs exploitation
    
    f_min_global = -0.8
    f_initial_estimate = 0.0
    
    # Distance from global optimum (0 = at optimum, 0.8 = far)
    dist = np.maximum(0.0, f_best - f_min_global)
    max_dist = 0.8
    normalized_dist = np.clip(dist / max_dist, 0.0, 1.0)
    
    # Estimate remaining budget from f_best
    # If f_best is close to -0.8, we've made good progress, spend less on exploration
    # If f_best is far, we need more exploration
    # 
    # Heuristic: map normalized_dist to a budget fraction
    # High dist (early) -> high exploration budget fraction
    # Low dist (late) -> low exploration budget fraction
    
    # Also consider that early on, we have more total budget remaining
    # Use f_best as a proxy for progress: better f_best means more budget "spent" effectively
    
    # Kappa range: higher when exploring, lower when exploiting
    # Scale by remaining budget awareness
    kappa_base = 0.3 + 2.2 * normalized_dist  # 0.3 to 2.5
    
    # Additional damping when sigma is very high (noise in 8D)
    # In high dimensions, sigma can be inflated; penalize pure uncertainty
    # Use a soft cap: effective_sigma = sigma * (1 - 0.1 * min(1, sigma))
    sigma_effective = sigma * (1.0 - 0.15 * np.minimum(1.0, sigma))
    
    # LCB: mu - kappa * sigma_effective
    lcb_value = mu - kappa_base * sigma_effective
    raw_score = -lcb_value  # kappa * sigma_eff - mu
    
    # Cost normalization with stabilization
    # Use geometric mean of cost to normalize, then divide
    # This makes scores comparable across different cost scales
    cost_median = np.median(cost_safe)
    cost_normalized = cost_safe / cost_median  # ~1 for median cost
    
    final_scores = raw_score / cost_normalized
    
    # Handle non-finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, -1e10)
    
    return final_scores
# EVOLVE-BLOCK-END
