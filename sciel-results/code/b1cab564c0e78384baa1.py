"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid approach combining expected improvement per unit cost
    with exploration bonus based on uncertainty.
    """
    # Track time to ensure we finish well within limits
    t_start = time.time()
    
    # Handle edge cases: if sigma is very small, we're confident
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-10)
    ei_per_cost = ei / cost_safe
    
    # Add exploration bonus: favor points with higher uncertainty
    # when we haven't found good values yet, and less when we have
    # This helps escape local optima
    max_sigma = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    exploration_bonus = 0.1 * sigma_safe / max_sigma if max_sigma > 0 else 0.0
    
    # Combine: primarily EI per cost, with a small exploration component
    scores = ei_per_cost + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check - if we're taking too long, return immediately
    # (This shouldn't happen for vectorized ops, but as safety)
    if time.time() - t_start > 250:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
