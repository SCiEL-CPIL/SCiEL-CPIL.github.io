"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute the Upper Confidence Bound (UCB) acquisition score for each candidate.
    
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
        UCB scores; higher means more promising.
    """
    # Ensure sigma is strictly positive to avoid numerical issues, though UCB doesn't divide by it
    sigma = np.maximum(sigma, 1e-12)
    
    # For minimization, we want to minimize mu.
    # We formulate an acquisition function to maximize:
    # Score = -mu + beta * sigma
    # This balances exploitation (-mu) and exploration (sigma).
    
    beta = 2.0
    
    ucb_scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, -np.inf)
    
    return ucb_scores
# EVOLVE-BLOCK-END
