import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Hartmann 6D).
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: scalar, the lowest observed value so far
    - cost: array of shape (2000,) with evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = evaluate next)
    """
    # Deadline guard: track time to avoid exceeding 300s per evaluation
    # (though this function is called many times, each call should be fast)
    
    # Ensure arrays are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma, Phi is CDF, phi is PDF of standard normal
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get value per unit cost
    # Higher EI per unit cost is preferred
    scores = ei / cost_safe
    
    # Add a small exploration bonus based on sigma to encourage exploring uncertain regions
    # This helps in high-dimensional spaces where EI alone may be too greedy
    # Scale the exploration bonus relative to EI magnitude
    ei_scale = np.median(ei) if np.any(ei > 0) else 1.0
    exploration_bonus = 0.1 * (sigma_safe / cost_safe) * (ei_scale / (np.median(sigma_safe) + 1e-10))
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
