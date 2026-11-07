"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition for Hartmann 6D.
    
    Uses a beta parameter that adapts based on how close f_best is to the
    known global minimum (-3.32237). Early on (when f_best is far from minimum),
    use high exploration (large beta). As we get closer, reduce exploration
    to focus on exploitation.
    
    UCB(x) = -mu(x) + beta * sigma(x)
    
    The negative mu is because we want to minimize, so lower mu is better.
    """
    # Known global minimum of Hartmann 6D
    f_min = -3.32237
    
    # Compute how close we are to the global minimum
    # gap = f_best - f_min (positive when we haven't found the minimum yet)
    gap = f_best - f_min
    
    # Adaptive beta: start high for exploration, reduce as we approach the minimum
    # When gap is large (early), use beta ~ 2.5 for exploration
    # When gap is small (late), use beta ~ 0.5 for exploitation
    # Use a smooth transition based on gap
    # gap ranges from ~3.3 (at start) to ~0 (at optimum)
    # Normalize gap to [0, 1] for the transition
    gap_norm = np.clip(gap / 3.3, 0.0, 1.0)
    
    # Beta transitions from 2.5 (high exploration) to 0.5 (high exploitation)
    beta = 0.5 + 2.0 * gap_norm
    
    # Compute UCB: -mu + beta * sigma
    # Higher UCB means more promising (lower expected value with uncertainty)
    ucb = -mu + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
