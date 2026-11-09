import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Lower Confidence Bound (LCB).
    
    Args:
        mu: Mean of the GP posterior (shape 2000,)
        sigma: Standard deviation of the GP posterior (shape 2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point (shape 2000,)
        
    Returns:
        Scores (shape 2000,), higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zero costs or invalid values
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls the exploration-exploitation trade-off
    # Higher beta -> more exploration
    beta = 2.0
    
    # Lower Confidence Bound for minimization:
    # We want to find the point with the lowest possible value.
    # LCB = mu - beta * sigma
    # We want to MINIMIZE LCB.
    # We also want to MINIMIZE cost.
    # We combine them by dividing LCB by cost? 
    # If LCB is negative (good), -LCB is positive.
    # Score = -LCB / cost = (beta * sigma - mu) / cost
    
    lcb = mu - beta * sigma
    scores = -lcb / cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
