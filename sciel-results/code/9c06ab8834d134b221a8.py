# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Samples a single scalar theta from N(mu, sigma^2) for each candidate, then scores
    using -theta / cost. This stochastic approach naturally balances exploration and
    exploitation without tuned hyperparameters.
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to reasonable ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 1.0, 2.0)
    
    # Thompson Sampling: sample theta ~ N(mu, sigma^2) for each candidate
    # Use a single random draw for all candidates to maintain correlation
    # Actually, we sample independently for each candidate
    rng = np.random.default_rng()
    theta = rng.normal(loc=mu, scale=sigma, size=mu.shape)
    
    # Score: negative cost-normalized value (lower theta is better, so -theta/cost is higher for better candidates)
    score = -theta / np.maximum(cost, 1.0)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
