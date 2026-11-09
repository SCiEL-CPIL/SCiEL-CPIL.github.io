"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0  # 10s safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines Expected Improvement with a
    diversity-aware bonus to encourage exploration of unexplored regions.
    
    mu, sigma: shape (2000,) in objective units
    f_best: lowest observed value
    Returns: shape (2000,) finite scores, higher = evaluate next
    """
    # Check deadline
    if not _elapsed_ok():
        # Return uniform scores to avoid bias if we're running out of time
        return np.random.rand(len(mu))
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative EI (clip small negatives from numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Diversity bonus: reward points where the GP is uncertain
    # This encourages exploration of regions with high sigma
    # Scale sigma to [0, 1] range for the bonus
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Combination: EI dominates, but add a moderate exploration bonus
    # Weight the exploration term to be significant but not dominant
    # Use a square root to make the exploration bonus grow more slowly
    exploration_bonus = 0.1 * np.sqrt(sigma_norm)
    
    score = ei + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
