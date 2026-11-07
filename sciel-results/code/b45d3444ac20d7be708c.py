"""Cost-aware acquisition using Expected Improvement (EI) with cost normalization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Expected Improvement (EI).
    
    Args:
        mu: Array of GP means, shape (2000,)
        sigma: Array of GP standard deviations, shape (2000,)
        f_best: Scalar, lowest observed value.
        cost: Array of costs for each candidate, shape (2000,)
        
    Returns:
        Array of scores, shape (2000,), higher is better.
    """
    epsilon = 1e-9
    
    # Handle zero or near-zero sigma
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Z-score: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - y, 0)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI divided by cost
    # This encourages finding improvements while being mindful of evaluation cost
    scores = ei / (cost + epsilon)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
