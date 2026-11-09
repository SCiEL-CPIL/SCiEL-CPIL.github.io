"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Branin 2D optimization.
    
    Uses UCB (Upper Confidence Bound) as the primary acquisition signal.
    For multi-modal functions like Branin, strong exploration via UCB is critical
    to find all basins. The beta parameter controls exploration vs exploitation.
    
    Score = -mu + beta * sigma (higher is better)
    """
    # Safety margin for sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Upper Confidence Bound for minimization
    # -mu rewards low predicted values, beta*sigma rewards high uncertainty
    # beta=2.0 provides strong exploration for the multi-modal Branin function
    beta = 2.0
    score = -mu + beta * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
