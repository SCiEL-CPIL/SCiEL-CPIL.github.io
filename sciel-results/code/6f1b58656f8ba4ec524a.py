"""Acquisition function for Bayesian optimization of Beale 2D using UCB."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (n,)
        UCB scores for each candidate. Higher scores indicate more promising candidates.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter: balance between exploitation and exploration
    # For minimization, we want to maximize: -mu + beta * sigma
    # This encourages exploration in uncertain regions while exploiting low mean values
    
    # Adaptive beta based on the scale of the problem
    # Use a fixed beta that works well for 2D optimization with 30 evaluations
    beta = 2.0
    
    # UCB for minimization: we want to find points likely to have low values
    # The upper bound on the function value is mu + beta * sigma
    # To minimize, we want to minimize this upper bound, so we maximize the negative
    ucb_scores = -mu + beta * sigma_safe
    
    return ucb_scores
# EVOLVE-BLOCK-END
