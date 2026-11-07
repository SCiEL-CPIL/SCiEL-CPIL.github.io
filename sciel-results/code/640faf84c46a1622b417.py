"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # wall-clock seconds total allowed


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses Probability of Improvement (PI) to focus on exploitation while
    still allowing exploration through the probabilistic framework.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # Higher probability means higher score
    score = norm.cdf(z)
    
    # Ensure finite output, replacing non-finite values with 0
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
