import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
_LIMIT = 300.0

def _min_max_normalize(arr):
    """Normalize array to [0, 1] range safely."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_range = arr_max - arr_min
    if arr_range < 1e-12:
        # If all values are the same, return uniform scores to avoid bias
        return np.full_like(arr, 0.5)
    return (arr - arr_min) / arr_range

def acquisition(mu, sigma, f_best):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _LIMIT - 5.0:
        # Fallback to greedy: pick lowest mu
        # Return scores such that lowest mu has highest score
        # Score = -mu
        return -mu

    # Ensure inputs are finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # For minimization, improvement is f_best - mu.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in extreme tails, though norm.cdf/pdf handle it well
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
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
    # If any non-finite values slipped through, replace with 0 (low priority)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
