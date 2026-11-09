import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    array of shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 300s limit
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Use Cost-Adjusted LCB (Lower Confidence Bound) for minimization
    # We want to minimize f(x). The LCB is mu - beta * sigma.
    # We want to select the point with the lowest LCB.
    # Therefore, we maximize -LCB / cost = (beta * sigma - mu) / cost.
    # This encourages points with low predicted mean and high uncertainty,
    # penalized by evaluation cost.
    
    beta = 2.0  # Exploration-exploitation trade-off parameter
    
    lcb = mu - beta * sigma
    score = -lcb / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time
    if time.time() > deadline:
        return score
    
    return score
