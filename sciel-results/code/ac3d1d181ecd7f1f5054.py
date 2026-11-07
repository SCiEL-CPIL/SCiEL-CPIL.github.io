"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Styblinski-Tang 1D using a
    UCB-style approach with moderate exploration for multimodal landscapes.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB acquisition: -mu + kappa * sigma
    # Higher scores mean: lower predicted mean (exploitation) and/or higher uncertainty (exploration)
    # For minimization, we want to maximize -mu + kappa * sigma
    kappa = 1.0
    score = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
