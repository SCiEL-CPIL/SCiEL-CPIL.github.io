"""Rewritten program using Probability of Improvement (PI) acquisition."""
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Probability of Improvement (PI).
    
    For minimization, PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    Args:
        mu: Array of mean predictions (shape: (2000,))
        sigma: Array of standard deviation predictions (shape: (2000,))
        f_best: The lowest observed value so far.
        
    Returns:
        Array of acquisition scores (shape: (2000,)), higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score: (f_best - mu) / sigma
    # Higher z means higher probability that f(x) < f_best
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: Phi(z)
    # This is in [0, 1], higher means more likely to improve
    pi = norm.cdf(z)
    
    # Ensure output is finite
    pi = np.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    
    return pi
