"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: we must return within the 300s limit per evaluation.
# We'll use a conservative internal deadline to ensure we don't get killed.
_DEADLINE = 300.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior
    sigma : ndarray, shape (2000,)
        Std of GP posterior
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Check time budget
    if _time_left() < 1.0:
        # Return a simple fallback if we're running out of time
        return np.zeros_like(mu)
    
    # Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Clip z to avoid numerical issues with very large values
    z = np.clip(z, -50, 50)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative and finite
    EI = np.maximum(EI, 0.0)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    # Cost-aware: divide by cost
    scores = EI / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
