import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_time_limit = 330.0  # Total wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with cost-awareness and adaptive exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _time_limit - 5.0:
        # Fallback: use PI as a simple, fast score
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        score = pi / np.maximum(cost, 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI per unit cost
    # Higher cost means less value per unit, so we divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
