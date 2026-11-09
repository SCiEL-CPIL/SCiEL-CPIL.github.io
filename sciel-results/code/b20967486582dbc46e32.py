"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for the deadline guard
_start_time = time.time()
_deadline = 330.0  # 330s wall clock limit

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    Draws a single sample from the posterior at each candidate point
    and returns the negative sample as the score (higher is better).
    
    Args:
    mu: array of GP means, shape (N,)
    sigma: array of GP std devs, shape (N,)
    f_best: current best observed value (scalar)
    
    Returns:
    scores: array of shape (N,), higher is better
    """
    # Deadline guard
    if time.time() - _start_time > _deadline - 1.0:
        # Fallback: return based on low mu (exploitation only) to be safe and fast
        return -mu

    # Ensure sigma is non-zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw a sample from N(mu, sigma^2) at each point
    # The sample represents a possible true function value at that point
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # For minimization, we want to pick the point with the lowest sample
    # So we return negative sample as the score (higher score = lower sample)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
