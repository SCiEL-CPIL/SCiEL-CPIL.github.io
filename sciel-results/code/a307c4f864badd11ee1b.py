"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_bo_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Thompson Sampling draws a sample from the posterior at each candidate point
    and scores candidates by the negative of this sample. This implicitly balances
    exploration and exploitation through stochastic sampling rather than a 
    deterministic confidence bound.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = more promising)
    """
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    # Deadline Guard: 300s limit, use 280s as safety margin
    if time.time() - _bo_start_time > 280:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu
    
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw a single random sample from the posterior
    # theta = mu + sigma * Z, where Z is standard normal noise
    # We score by -theta (since we want to minimize, lower theta is better,
    # so higher score means more promising)
    Z = np.random.standard_normal(len(mu))
    theta = mu + sigma * Z
    score = -theta
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
