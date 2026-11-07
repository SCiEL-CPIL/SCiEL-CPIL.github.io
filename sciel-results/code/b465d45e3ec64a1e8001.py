import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian Optimization.
    Uses Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: Array of shape (2000,) with mean predictions.
        sigma: Array of shape (2000,) with standard deviation predictions.
        f_best: The lowest observed value so far.
        
    Returns:
        Array of shape (2000,) with scores. Higher score means more promising.
    """
    # UCB for minimization: Maximize -mu + beta * sigma
    # This prioritizes points with low mean (exploitation) 
    # and high uncertainty (exploration).
    
    # Handle edge cases where sigma is 0 to avoid numerical issues, 
    # though UCB doesn't divide by sigma, so it's safer than EI/PI.
    # However, if sigma is 0, we just rely on -mu.
    
    beta = 2.0
    
    # Calculate UCB scores
    # We want to minimize f(x), so we maximize the score.
    # Score = -mu(x) + beta * sigma(x)
    scores = -mu + beta * sigma
    
    return scores
