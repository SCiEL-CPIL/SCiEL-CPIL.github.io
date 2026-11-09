"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Probability of Improvement (PI) normalized by cost^0.5.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Convert to numpy arrays for efficient computation
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive (avoid division by zero)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for probability of improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize by cost^0.5 to balance exploration efficiency against computational budget
    scores = pi / np.sqrt(cost)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
