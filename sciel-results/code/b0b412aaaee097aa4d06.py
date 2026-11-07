"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Use Upper Confidence Bound (UCB) with an adaptive exploration term.
    
    The UCB acquisition balances exploration and exploitation by adding
    a penalty term that encourages exploration in regions with high uncertainty.
    We use a time-adaptive kappa that starts large and decays, but since we don't
    have explicit time step info, we use a fixed reasonable value.
    
    For minimization, we want to maximize: -mu + kappa * sigma
    This is equivalent to UCB for minimization.
    
    We also add a small regularization to avoid numerical issues.
    """
    # Clamp sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-8)
    
    # Use UCB for minimization: we want to find points where -f is high,
    # i.e., where f is low. The UCB for the negative objective would be:
    # -mu + kappa * sigma
    # 
    # We choose kappa to balance exploration and exploitation.
    # A common choice is around 1.0-2.0 for well-scaled problems.
    # Since our objective is sphere function on [-5,5], and GP is standardized,
    # we use a moderate exploration coefficient.
    
    kappa = 1.5
    
    # UCB score for minimization: higher score means more promising
    # We negate mu because we're minimizing
    scores = -mu + kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
