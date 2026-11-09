"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization on the three-hump camel 2D.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions for 2000 candidates, in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions for 2000 candidates, in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of coordinates scaled to [0,1]).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred. Must be finite.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute expected improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Use numerical stability: for large z, Phi(z) ~ 1 and phi(z) ~ 0
    # For very negative z, EI ~ 0
    ei = np.zeros_like(mu)
    
    # Standard EI computation
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-12)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
