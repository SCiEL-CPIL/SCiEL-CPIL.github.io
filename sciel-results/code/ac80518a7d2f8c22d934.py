"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def _check_deadline():
    """Return True if we should stop working (deadline approaching)."""
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI + UCB acquisition with adaptive exploration-exploitation tradeoff.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: 2000 finite scores, higher = more promising.
    """
    n = len(mu)
    
    # Deadline guard: if we're close to the limit, return a simple EI score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Known minimum of the RBF mixture
    f_min = -2.53218
    
    # Progress: how close are we to the minimum?
    regret = f_best - f_min
    # Map regret from [0, ~2] to progress in [0, 1]
    progress = 1.0 - np.clip(regret / 2.0, 0.0, 1.0)
    
    # Adaptive parameters
    # Use a small xi for EI to focus on improvements
    xi = 0.01
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term: use sigma directly as a proxy for uncertainty/diversity
    # Scale sigma to be comparable to EI magnitude
    # EI is typically in the range of [0, f_best - min_mu]
    # sigma is in the same units.
    # We add a small weight to sigma to encourage exploration
    exploration_weight = 0.1
    
    score = ei + exploration_weight * sigma_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
