"""Rewritten program using LCB acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
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
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and has a small floor to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want to minimize, so we look for low mu relative to uncertainty.
    # The standard LCB for minimization is mu - kappa * sigma.
    # We choose kappa to balance exploration and exploitation.
    # For multimodal functions like Styblinski-Tang, a higher kappa encourages exploration
    # to escape local minima.
    kappa = 2.0
    
    scores = mu - kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
