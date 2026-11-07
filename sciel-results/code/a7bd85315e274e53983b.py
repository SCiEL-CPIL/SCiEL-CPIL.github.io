import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) strategy.
    
    This is a structural change from Expected Improvement (EI).
    LCB directly maximizes the lower bound of the function value,
    encouraging exploration of regions with high uncertainty (high sigma)
    and low predicted mean (low mu).
    
    Score = -mu + beta * sigma
    """
    # Heuristic for beta: 
    # A value of 2.5 provides a strong balance between exploration and exploitation,
    # suitable for the small evaluation budget of 30.
    beta = 2.5
    
    # Calculate LCB scores
    # We want to minimize the function, so we look for the lowest lower bound.
    # Lower bound = mu - beta * sigma
    # To maximize the score (higher is better for selection), we negate the lower bound:
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    scores = -mu + beta * sigma
    
    return scores
