"""Cost-aware acquisition function using Probability of Improvement (PI) with quadratic cost penalty."""
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 290  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Computes a cost-aware acquisition score for each candidate.
    
    Parameters:
    - mu: array of shape (N,) containing GP mean predictions.
    - sigma: array of shape (N,) containing GP standard deviations.
    - f_best: float, the best function value observed so far.
    - cost: array of shape (N,) containing the cost of evaluating each candidate.
    
    Returns:
    - scores: array of shape (N,) containing the acquisition scores.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        # Return a valid, finite array to avoid invalidation.
        # A constant score will lead to deterministic selection (likely index 0).
        return np.zeros_like(mu)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using standard normal CDF
    pi = norm.cdf(z)
    
    # Cost-aware penalty
    # Cost is in [1, 2]. 
    # Using cost^2 amplifies the preference for cheaper points (factor of 4 between min and max cost).
    cost_safe = np.maximum(cost, 1.0)
    cost_penalty = cost_safe ** 2
    
    # Acquisition Score: Higher is better
    # We want high PI and low cost.
    scores = pi / cost_penalty
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
