"""Rewritten acquisition: Probability of Improvement (PI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute the Probability of Improvement (PI) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (N,) - mean of the GP posterior at candidate points
    sigma: array of shape (N,) - standard deviation of the GP posterior at candidate points
    f_best: float - the best (lowest) observed function value so far
    
    Returns:
    scores: array of shape (N,) - PI values, higher means more promising
    """
    # Ensure sigma is positive to avoid division by zero
    # Use a small epsilon to handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: (f_best - mu) / sigma
    # This represents how many standard deviations f_best is above mu
    z = (f_best - mu) / sigma_safe
    
    # PI is the probability that f(x) < f_best
    # P(f(x) < f_best) = P(mu + sigma * Z < f_best) = P(Z < (f_best - mu) / sigma)
    # where Z is a standard normal variable
    pi_scores = norm.cdf(z)
    
    # Handle numerical issues
    scores = np.where(np.isfinite(pi_scores), pi_scores, 0.0)
    
    # PI values are in [0, 1]
    return scores
# EVOLVE-BLOCK-END
