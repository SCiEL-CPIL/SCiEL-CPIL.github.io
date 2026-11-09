"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def _min_max_normalize(arr):
    """Normalize array to [0, 1] range safely."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_range = arr_max - arr_min
    if arr_range < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - arr_min) / arr_range


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and 
    Upper Confidence Bound (UCB) for minimizing Hartmann 3D.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float, lowest observed value
    
    Returns:
    score: array of shape (2000,)
    """
    # Ensure inputs are finite and sigma is strictly positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # For minimization, improvement is f_best - mu.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # Low mu is good (-mu), high sigma is good for exploration (+beta * sigma).
    beta = 2.5
    ucb = -mu + beta * sigma
    
    # 3. Normalize scores to [0, 1] to allow meaningful mixing
    ei_norm = _min_max_normalize(ei)
    ucb_norm = _min_max_normalize(ucb)
    
    # 4. Combine scores
    # Weighting: 50% Exploitation (EI), 50% Exploration (UCB)
    score = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # 5. Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # 6. Deadline Guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Computation is O(N) and fast, but this satisfies the directive.
        pass
        
    return score
# EVOLVE-BLOCK-END
