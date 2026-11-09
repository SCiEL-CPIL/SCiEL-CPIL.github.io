import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Upper Confidence Bound (UCB) for minimization.
    """
    # Exploration coefficient
    beta = 2.5
    
    # Clip sigma to avoid negative values due to numerical precision
    sigma = np.clip(sigma, 0, None)
    
    # UCB score for minimization:
    # We want to find the point with the lowest potential value.
    # Potential value estimate: mu - beta * sigma
    # We return a score to be MAXIMIZED, so we negate the potential value.
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
