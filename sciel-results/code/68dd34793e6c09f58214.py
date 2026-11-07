"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition for minimization.
    
    Parameters
    ----------
    mu : ndarray, shape (n_candidates,)
        Posterior mean at candidate points (in objective units).
    sigma : ndarray, shape (n_candidates,)
        Posterior standard deviation at candidate points (in objective units).
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want low mu and high sigma
    # The LCB value is mu - k * sigma
    # We return the negative of LCB so that higher scores mean "more promising"
    # (since we want to minimize, a lower LCB is better, so we negate it)
    
    # Use a moderate exploration constant
    # k=3.0 is aggressive but reasonable for a 2D problem with isolated minimum
    k = 3.0
    
    # Compute LCB: mu - k * sigma
    lcb = mu - k * sigma_safe
    
    # Return -lcb so that higher scores correspond to lower LCB values
    # (i.e., more promising for minimization)
    scores = -lcb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
