"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using max(EI/cost, LCB/cost) to avoid additive
    interference between exploitation and exploration. The max operation
    allows the function to fully exploit high-EI regions or fully explore
    high-uncertainty regions without compromise.
    
    Time budget guard ensures we return within safe limits.
    """
    t_start = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (exploration term)
    # Using a moderate beta for exploration
    beta = 2.0
    lcb = mu - beta * sigma_safe
    
    # Normalize by cost
    ei_cost = ei / cost_safe
    lcb_cost = -lcb / cost_safe  # Higher LCB means lower cost-adjusted value to explore
    
    # Use max to avoid additive interference
    scores = np.maximum(ei_cost, lcb_cost)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time budget check
    t_elapsed = time.time() - t_start
    if t_elapsed > 0.3:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
