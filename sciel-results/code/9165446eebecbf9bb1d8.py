"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline checking
_START_TIME = time.time()
_DEADLINE = 250.0  # Safety margin before the 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian Optimization.
    
    Uses Thompson Sampling to naturally balance exploration and exploitation
    without tuned hyperparameters. Samples theta_i ~ N(mu_i, sigma_i) for each
    candidate, then scores using -theta_i / cost_i to prioritize candidates
    with low expected objective value and low evaluation cost.
    
    For multimodal landscapes like Styblinski-Tang, the stochastic nature of
    Thompson Sampling provides robust exploration that can escape local minima
    more effectively than deterministic acquisition functions.
    """
    # Check deadline - if we're running out of time, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return simple cost-normalized negative mu
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_safe, 1e-8)
        fallback = -mu_safe / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, -np.inf)
        return fallback
    
    # Convert inputs to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample theta_i ~ N(mu_i, sigma_i) for each candidate
    # This provides a natural exploration-exploitation balance
    theta = mu + sigma * np.random.standard_normal(len(mu))
    
    # Score: higher is better. We want low theta (good objective) and low cost.
    # Score = -theta / cost
    scores = -theta / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
