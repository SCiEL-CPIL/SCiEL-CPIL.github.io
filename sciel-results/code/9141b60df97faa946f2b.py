"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = time.time() + 325.0


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses Probability of Improvement (PoI) which decouples exploration
    from improvement magnitude, favoring broad coverage of Branin's
    multiple local minima over EI's exploitation bias.
    
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior std
    f_best: best (lowest) observed value so far
    
    Returns: array of shape (2000,) with scores (higher = more promising)
    """
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    # z = (f_best - mu) / sigma
    # PoI = Phi(z)
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Ensure finite values
    poi = np.where(np.isfinite(poi), poi, 0.0)
    
    return poi
# EVOLVE-BLOCK-END
