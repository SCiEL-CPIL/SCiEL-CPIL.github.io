"""Rewritten program using LCB acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    Returns scores to be MAXIMIZED.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
        All values are finite.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and has a small floor to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB for minimization is mu - kappa * sigma.
    # We want to MINIMIZE the LCB.
    # The interface asks for scores to be MAXIMIZED.
    # So we return -LCB = kappa * sigma - mu.
    
    # kappa = 2.0 encourages exploration more than 1.0
    kappa = 2.0
    
    scores = kappa * sigma - mu
    
    # Ensure all scores are finite
    # If any score is non-finite (nan/inf), replace with a very low finite value
    finite_mask = np.isfinite(scores)
    if not np.all(finite_mask):
        if np.any(finite_mask):
            min_finite = np.min(scores[finite_mask])
        else:
            min_finite = -1e10
        scores[~finite_mask] = min_finite - 1.0
        
    return scores
# EVOLVE-BLOCK-END
