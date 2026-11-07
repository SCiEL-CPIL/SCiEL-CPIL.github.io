import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an exploration bonus.
    
    Parameters:
    - mu: array of shape (2000,) - posterior mean
    - sigma: array of shape (2000,) - posterior standard deviation
    - f_best: float - best observed value so far
    
    Returns:
    - array of shape (2000,) - scores, higher is better
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component for exploration
    ucb = -mu - 1.5 * sigma_safe  # negative because we minimize
    
    # Blend: primarily EI with some UCB exploration
    # Use a fixed blend that works well for early-to-mid optimization
    score = 0.8 * ei + 0.2 * ucb
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
