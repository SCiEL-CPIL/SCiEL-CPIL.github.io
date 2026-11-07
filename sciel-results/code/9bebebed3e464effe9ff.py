"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for GP hyperparameters to avoid recomputation
_gp_cache = {}

def _compute_gp_params(mu, sigma, f_best, cost):
    """Compute GP-based parameters with caching."""
    cache_key = (id(mu), id(sigma), id(f_best), id(cost), mu.shape)
    if cache_key in _gp_cache:
        return _gp_cache[cache_key]
    
    # Use Expected Improvement (EI) with cost awareness
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This is a more principled approach than PI
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Standard EI
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    # Use a temperature parameter to balance exploration vs exploitation
    score = ei / cost_safe
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    _gp_cache[cache_key] = score
    return score


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function using Expected Improvement."""
    # Clear cache periodically to avoid memory issues
    if len(_gp_cache) > 100:
        _gp_cache.clear()
    
    return _compute_gp_params(mu, sigma, f_best, cost)
# EVOLVE-BLOCK-END
