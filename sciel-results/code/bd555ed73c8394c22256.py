"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile-based Probability of Improvement (POI) acquisition function.
    
    Instead of Expected Improvement (which weights by magnitude of improvement),
    this ranks candidates purely by their probability of beating the incumbent:
    P(f < f_best) = Phi((f_best - mu) / sigma).
    
    This decouples selection pressure from the absolute scale of improvement,
    focusing exploration on regions where there is a high likelihood of finding
    a better point, regardless of how much better it might be.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be selected next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero; use a small epsilon
    eps = 1e-9
    safe_sigma = np.maximum(sigma, eps)
    
    # Standard normal CDF: P(f < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / safe_sigma
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
