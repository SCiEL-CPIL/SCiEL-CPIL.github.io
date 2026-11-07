"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track global time for deadline guard
_start_time = None
_DEADLINE_SECONDS = 325.0  # 5s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Probability of Improvement (PI) as the structural change from EI.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running out of time, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        return -np.asarray(mu, dtype=np.float64).copy()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    # Using a small floor to maintain numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI)
    # PI = Phi((f_best - mu) / sigma)
    # This favors points with high likelihood of surpassing current best,
    # providing a distinct exploration bias compared to EI
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
