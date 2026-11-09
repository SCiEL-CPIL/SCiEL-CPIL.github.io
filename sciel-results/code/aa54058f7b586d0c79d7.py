"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with cost normalization and an exploration bonus.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on sigma to encourage exploration
    # This helps when EI is small but uncertainty is high
    exploration_bonus = 0.1 * sigma
    
    # Combine EI and exploration bonus
    combined = ei + exploration_bonus
    
    # Normalize by cost (cost-aware)
    scores = combined / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
