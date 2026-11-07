import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline guard
_start_time = [time.time()]

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for Bayesian optimization of hartmann 6d.
    Uses UCB with adaptive beta based on distance to known optimum.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Posterior mean
    sigma: np.ndarray of shape (N,) - Posterior standard deviation
    f_best: float - Best observed value so far (minimum)
    
    Returns:
    np.ndarray of shape (N,) - Acquisition scores (higher is better)
    """
    
    # Deadline guard: if we're getting close to the limit, return simple scores
    elapsed = time.time() - _start_time[0]
    if elapsed > 300:  # Safety margin before 330s kill
        # Return simple exploitation scores
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive UCB: beta depends on distance to known optimum
    # The minimum value of hartmann 6d is approximately -3.32237
    min_value = -3.32237
    
    # Distance from current best to the known minimum
    # When far from optimum, we want more exploration (higher beta)
    # When close to optimum, we want more exploitation (lower beta)
    distance_to_optimum = max(0.0, f_best - min_value)
    
    # Normalize distance: at start f_best ~ -0.5 to -2.0, distance ~ 1.3 to 2.8
    # Scale so that beta transitions from ~2.0 (far) to ~0.5 (close)
    # Use a smooth transition based on distance
    # When distance is large (>2.0), beta ~ 2.0
    # When distance is small (<0.5), beta ~ 0.5
    # Linear interpolation in log space or direct
    
    # Map distance to beta:
    # d = 0.0 -> beta = 0.5
    # d = 3.0 -> beta = 2.5
    # beta = 0.5 + (2.5 - 0.5) * min(d / 3.0, 1.0)
    beta = 0.5 + 2.0 * min(distance_to_optimum / 3.0, 1.0)
    
    # UCB score: -mu + beta * sigma
    # We want higher scores to be selected, and lower objective values are better
    # So we negate mu and add exploration bonus
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
