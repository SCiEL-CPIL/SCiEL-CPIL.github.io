"""Program with UCB (Upper Confidence Bound) acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    We maximize the score: beta * sigma - mu.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best (lowest) observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB parameter for exploration
    # A value of 2.5 balances exploration and exploitation for this problem size
    beta = 2.5
    
    # For minimization, we want to maximize the lower bound of the confidence interval?
    # No, we want to find the minimum.
    # The lower bound is L(x) = mu(x) - beta * sigma(x).
    # We want to minimize L(x).
    # So we maximize -L(x) = beta * sigma(x) - mu(x).
    scores = beta * sigma_safe - mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
