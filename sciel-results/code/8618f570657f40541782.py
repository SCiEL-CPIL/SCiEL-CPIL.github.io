"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Minimum-variance acquisition: score candidates solely by posterior standard deviation.
    This forces the GP to resolve uncertainty in unexplored regions before exploiting,
    preventing premature convergence to local noise minima.
    
    For sphere 1d on [-5,5], the global minimum is at x=0 with f=0.
    A pure exploration strategy (maximize sigma) will systematically probe
    unexplored regions, eventually finding the neighborhood of x=0.
    """
    # Deadline guard: if we're close to the limit, return a safe default
    if time.time() > _DEADLINE:
        # Return a neutral score that picks the first candidate
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Score is simply the posterior standard deviation.
    # Higher sigma means more uncertainty, so we explore there.
    # This is a deterministic minimum-variance acquisition.
    score = sigma.copy()
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
