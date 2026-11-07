"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a cost-aware modification.
    
    EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    
    This is a well-known acquisition function that balances exploitation and exploration.
    We modify it to account for evaluation cost by dividing by cost.
    
    Additionally, we add a small exploration bonus based on sigma to encourage
    visiting uncertain regions, but weighted by the cost.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > 310.0:
        # Fallback: simple sigma-based score normalized by cost
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        fallback = np.maximum(sigma_arr, 1e-8) / np.maximum(cost_arr, 1e-8)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware modification: divide by cost
    # This penalizes expensive evaluations
    EI_cost_aware = EI / cost_safe
    
    # Add exploration bonus based on sigma
    # This encourages visiting uncertain regions
    exploration_bonus = sigma_safe / cost_safe
    
    # Combine EI and exploration bonus
    scores = EI_cost_aware + 0.1 * exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
