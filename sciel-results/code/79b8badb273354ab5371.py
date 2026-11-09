"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 15.0  # seconds of safety margin
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: score is the negative of a single sample
    from the GP posterior at each candidate point. This implicitly balances
    exploration and exploitation without tuning a beta parameter.
    
    Parameters:
    mu: array-like, shape (N,), mean of the posterior distribution
    sigma: array-like, shape (N,), standard deviation of the posterior distribution
    f_best: float, the best (lowest) observed function value so far
    
    Returns:
    scores: numpy array, shape (N,), higher values indicate better candidates
    """
    global _call_count
    
    # Check deadline
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Return a simple UCB score as fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        sigma = np.maximum(sigma, 1e-9)
        scores = -mu + 2.0 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    _call_count += 1
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw a single sample from the posterior at each candidate
    # f_i ~ N(mu_i, sigma_i^2)
    # Score is -f_i (higher is better for minimization)
    sample = mu + sigma * np.random.randn(len(mu))
    
    # The acquisition score is the negative of the sampled value
    # This naturally balances exploration (high sigma -> more variance) 
    # and exploitation (low mu -> lower sampled value)
    scores = -sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
