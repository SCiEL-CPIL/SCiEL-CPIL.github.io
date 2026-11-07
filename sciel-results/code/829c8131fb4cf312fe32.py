"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (POI) for minimization.
    
    Args:
        mu: Posterior mean, shape (N,)
        sigma: Posterior standard deviation, shape (N,)
        f_best: Best observed value so far.
        
    Returns:
        Scores, shape (N,). Higher is better.
    """
    # Ensure sigma is non-negative to avoid division by zero or negative sqrt issues
    # Although sigma from GP should be positive, numerical errors can occur.
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive, probability of improvement is high.
    # If mu > f_best, z is negative, probability of improvement is low.
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    scores = norm.cdf(z)
    
    # Ensure scores are finite and within [0, 1]
    scores = np.clip(scores, 0.0, 1.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
