# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    Uses Upper Confidence Bound (UCB) with a fixed beta parameter.
    
    Args:
        mu: array of shape (2000,), posterior mean at candidate points
        sigma: array of shape (2000,), posterior standard deviation at candidate points
        f_best: lowest observed value so far
    
    Returns:
        array of shape (2000,) with scores, higher meaning evaluate next
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robust sigma floor to avoid numerical issues
    sigma_floor = 1e-8
    sigma_safe = np.maximum(sigma, sigma_floor)
    
    # UCB parameter: beta controls the exploration-exploitation tradeoff.
    # A higher beta encourages more exploration.
    # For Hartmann 3D with 30 evaluations, a moderate beta is often effective.
    # EI is equivalent to a specific non-linear combination of mu and sigma.
    # UCB is linear: Score = -mu + beta * sigma.
    # We want to minimize f, so we maximize the score.
    # If mu is low (good) and sigma is high (uncertain), the score should be high.
    # Score = -mu + beta * sigma
    
    # A common heuristic for beta in UCB for BO is around 1-3.
    # Let's try beta = 2.0. This is a strong exploration bonus.
    beta = 2.0
    
    # Calculate UCB score
    # We negate mu because we are minimizing, and we add the exploration term.
    ucb_scores = -mu + beta * sigma_safe
    
    # Ensure finite
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, -np.inf)
    
    return ucb_scores
# EVOLVE-BLOCK-END
