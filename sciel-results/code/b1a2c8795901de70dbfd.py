"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 280s safety margin (limit is 300s)


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples a single theta from N(mu, sigma^2) for each candidate,
    then scores using -theta / cost (higher = better, i.e., lower predicted
    value per unit cost is preferred).
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: prefer low cost and low mu
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Theoretical minimum for Styblinski-Tang 2D
    f_min_theoretical = -78.3323
    
    # Heuristic start value for progress calculation
    f_start = 20.0
    
    # Calculate progress in [0, 1]
    denom = f_start - f_min_theoretical
    progress = np.clip((f_start - f_best) / denom, 0.0, 1.0)
    
    # Thompson Sampling: sample theta ~ N(mu, sigma^2) for each candidate
    # Use a single random draw per candidate to introduce stochasticity
    # that naturally balances exploration and exploitation
    rng = np.random.default_rng()
    theta = mu + sigma * rng.standard_normal(len(mu))
    
    # Score: higher is better. We want low theta (low predicted value)
    # and low cost. So score = -theta / cost
    score = -theta / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
