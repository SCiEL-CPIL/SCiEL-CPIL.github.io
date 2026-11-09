import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Check if we're approaching the deadline (330s limit, use 325s as safety margin)
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a simple fallback to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        safe_cost = np.maximum(cost, 1e-10)
        return -mu / safe_cost
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Small epsilon to prevent division by zero
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / safe_sigma
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Cost-Adjusted EI
    c_ei = ei / safe_cost
    
    # Upper Confidence Bound for minimization
    # We want high scores for low mu and high uncertainty
    # UCB = -mu + beta * sigma
    # Use adaptive beta based on the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    beta = 1.5 + 0.5 * (sigma_std / (sigma_mean + eps)) if sigma_mean > eps else 1.5
    beta = np.clip(beta, 1.0, 3.0)
    
    ucb = -mu + beta * safe_sigma
    c_ucb = ucb / safe_cost
    
    # Lower Confidence Bound component (favor regions with low predicted values)
    lcb = -mu - beta * safe_sigma
    c_lcb = -lcb / safe_cost  # negate to make higher = better (lower mu is better)
    
    # Combine: use max of cost-adjusted EI and UCB
    # Also add a small exploration bonus for regions with high uncertainty
    uncertainty_bonus = 0.1 * safe_sigma / safe_cost
    
    scores = np.maximum(c_ei, c_ucb) + uncertainty_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
