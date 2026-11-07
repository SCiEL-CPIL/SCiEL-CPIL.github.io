"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement with adaptive UCB exploration,
    using a non-linear cost penalty to focus on cheaper regions.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) - in objective units
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive beta: higher when uncertainty is large relative to improvement
    # This encourages more exploration when we're not seeing much improvement
    improvement = np.maximum(f_best - mu, 0)
    beta = 0.5 * np.exp(-improvement / (sigma_safe + 1e-9))
    beta = np.clip(beta, 0.1, 1.0)
    
    # UCB-like exploration term
    ucb = -mu + beta * sigma_safe
    
    # Combine: primarily EI with some UCB
    raw_score = 0.8 * ei + 0.2 * ucb
    
    # Non-linear cost penalty: normalize cost to [0, 1] relative to current batch
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost_safe - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Quadratic cost penalty: more aggressive than linear division
    # Cheapest points get penalty ~1.0, most expensive get penalty ~0.33
    cost_penalty = 1.0 / (1.0 + 2.0 * cost_norm ** 1.5)
    
    # Apply cost penalty to the combined score
    score = raw_score * cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    if time.time() - t0 > 280:
        # Return simple cost-normalized EI as fallback
        fallback = ei / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
# EVOLVE-BLOCK-END
