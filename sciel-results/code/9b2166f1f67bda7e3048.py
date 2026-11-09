"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Compute the Lower Confidence Bound (LCB) acquisition score for each candidate.
    
    Parameters
    ----------
    mu : np.ndarray, shape (n_candidates,)
        Posterior mean of the objective at each candidate.
    sigma : np.ndarray, shape (n_candidates,)
        Posterior standard deviation of the objective at each candidate.
    f_best : float
        Best observed value so far (lowest, since we minimize).
    
    Returns
    -------
    np.ndarray, shape (n_candidates,)
        LCB scores; higher means more promising.
        Note: LCB is typically minimized, so we return -LCB to maximize.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we return -LCB = -mu + beta * sigma
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
