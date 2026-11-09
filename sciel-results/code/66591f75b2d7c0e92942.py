"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a combination of Expected Improvement (EI) and a small exploration bonus
    based on uncertainty to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps in early stages when EI might be dominated by noise
    exploration_bonus = 0.01 * sigma
    
    # Combine EI with exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
