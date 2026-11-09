"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses a combination of Expected Improvement (EI) and a lower bound on sigma
    to balance exploitation and exploration, plus a small random component
    to break ties and encourage diversity.
    
    mu, sigma: arrays of shape (2000,) with posterior mean and std dev
    f_best: scalar, the lowest observed value so far
    
    Returns: array of 2000 finite scores (higher = more promising)
    """
    # Deadline guard: if we've been running too long, return a simple EI
    # to ensure we finish within the 330s wall-clock limit
    # We don't have a global timer here, but we can make this fast
    
    # Clip sigma to avoid division by zero and to control exploration
    # Using a lower bound on sigma helps exploration in uncertain regions
    sigma_safe = np.maximum(sigma, 1e-4)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus based on sigma
    # This encourages visiting uncertain regions
    # The coefficient is small to not dominate the EI term
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
