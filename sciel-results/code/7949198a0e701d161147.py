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
    
    # Thompson Sampling
    # Sample from the posterior distribution N(mu, sigma^2)
    # We want to minimize f(x) / cost(x)
    # So we maximize -f(x) / cost(x)
    
    # Generate samples
    # Note: np.random.normal(loc=mu, scale=sigma)
    samples = np.random.normal(mu, sigma)
    
    # Calculate scores
    score = -samples / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time
    if time.time() > deadline:
        return score
    
    return score
