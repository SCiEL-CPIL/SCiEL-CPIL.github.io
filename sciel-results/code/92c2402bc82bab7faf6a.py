"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
# Set deadline to 320s to be safe against the 330s kill
_DEADLINE = _START_TIME + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Probability of Improvement (PI).
    
    Parameters:
    -----------
    mu : numpy.ndarray
        GP mean predictions, shape (2000,).
    sigma : numpy.ndarray
        GP standard deviations, shape (2000,).
    f_best : float
        Best observed function value so far.
    cost : numpy.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : numpy.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return zeros to indicate no preference, or random? 
        # Zeros are finite and valid.
        return np.zeros_like(mu)

    # Ensure sigma is positive to avoid division by zero
    # Using a small epsilon
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-scores for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, and PI is high.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is the CDF of the standard normal distribution at z.
    pi_scores = norm.cdf(z)
    
    # Cost-aware adjustment
    # We divide by cost to penalize expensive evaluations.
    # Cost is in [1, 2], so this is a mild penalty.
    cost_safe = np.maximum(cost, eps)
    
    scores = pi_scores / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
