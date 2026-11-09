# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Quantile-based Probability of Improvement.
    Maximizes P(f(x) < f_best - epsilon).
    This is equivalent to maximizing the CDF of the standard normal at
    z = (f_best - epsilon - mu) / sigma.
    """
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)

    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Define epsilon as a fraction of the current best value
    # This creates a margin for improvement
    epsilon = 0.1 * abs(f_best)
    
    # Calculate the z-score
    # z = (f_best - epsilon - mu) / sigma
    z_scores = (f_best - epsilon - mu) / sigma_safe
    
    # Compute the probability (CDF of standard normal)
    # This creates a bounded score in [0, 1]
    scores = norm.cdf(z_scores)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
