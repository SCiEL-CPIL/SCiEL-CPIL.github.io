"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a blend of Expected Improvement (EI) and Probability of Improvement (PI),
    normalized by cost^0.5 to balance exploration efficiency.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points (in objective units)
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points (in objective units)
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Clamp sigma to be positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Clamp cost to be positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI/PI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Blend EI and PI with adaptive weighting
    # When uncertainty is high relative to improvement, weight PI more
    # This provides more stable gradients in multimodal landscapes
    improvement = np.maximum(f_best - mu, 0)
    uncertainty = sigma
    
    # Adaptive weight: higher when uncertainty dominates
    weight = uncertainty / (improvement + uncertainty + 1e-9)
    
    # Combined acquisition score
    combined = (1 - weight) * ei + weight * pi
    
    # Normalize by cost^0.5 for better exploration balance
    scores = combined / np.sqrt(cost)
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
