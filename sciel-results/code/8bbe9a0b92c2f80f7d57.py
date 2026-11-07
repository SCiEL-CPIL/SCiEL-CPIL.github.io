"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Lower Quantile (LCB) approach for minimization.
    
    This targets the lower tail of the posterior predictive distribution,
    approximating the expected minimum of a large number of samples.
    Score = - (mu - z_{1-alpha} * sigma) = -mu + z_{1-alpha} * sigma.
    """
    start_time = time.time()
    deadline = start_time + 2.0 # 2s safety margin for a fast function
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid issues
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Use a very small alpha to target the lower bound (exploration)
    # z_{1-alpha} is the z-score such that P(Z < z) = 1 - alpha
    # For alpha = 1e-4, z is approx 3.72
    alpha = 1e-4
    beta = norm.ppf(1.0 - alpha)
    
    # Score = -mu + beta * sigma
    # Higher score is better (lower mu, higher sigma)
    score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
