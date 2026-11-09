"""Rewritten program for Bayesian optimization of Hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time
    
    
def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
        
    Uses Expected Improvement (EI) with an additional exploration term
    proportional to the posterior standard deviation.
        
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    start_time = time.time()
        
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
        
    # Ensure sigma is non-negative and add small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
        
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        
    # Add an exploration term proportional to uncertainty
    # This encourages visiting areas with high sigma
    exploration_weight = 0.5
    scores = ei + exploration_weight * sigma_safe
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
        
    # Explicit deadline guard
    if time.time() - start_time > 300:
        return scores
            
    return scores
# EVOLVE-BLOCK-END
