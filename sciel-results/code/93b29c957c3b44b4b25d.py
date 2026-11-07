"""Rewritten program for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (POI).
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    start_time = time.time()
    deadline = start_time + 280.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if len(mu) == 0:
        return np.array([])
        
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Probability of Improvement: P(f(x) < f_best)
    # z = (f_best - mu) / sigma
    # POI = Phi(z)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Ensure finite values
    scores = np.where(np.isfinite(poi), poi, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
