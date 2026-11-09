"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np


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
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-12)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # The UCB value for the objective is mu + beta * sigma.
    # To maximize the acquisition score (where higher is better for selection),
    # we use the negative of the objective UCB, or simply:
    # Score = -mu + beta * sigma
    # This encourages low mu (exploitation) and high sigma (exploration).
    
    beta = 2.0
    ucb_scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, 0.0)
    
    return ucb_scores
# EVOLVE-BLOCK-END
