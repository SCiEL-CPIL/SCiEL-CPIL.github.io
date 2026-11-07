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
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF of standard normal, phi is PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small exploration bonus: beta * sigma / cost
    # This encourages visiting uncertain regions, but penalized by cost
    beta = 0.1
    exploration_bonus = beta * sigma_safe / cost_safe
    
    # Combine EI and exploration bonus, then divide by cost for cost-awareness
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
