"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of Probability of Improvement (PI)
    and Expected Improvement (EI), normalized by cost to the power 0.5.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for improvement
    # Higher z means more likely to improve
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Blend PI and EI with adaptive weighting
    # When f_best is close to 0 (good), weight more on PI for stability
    # When f_best is large (early), weight more on EI for exploitation
    # Use a smooth transition based on f_best
    # Scale: f_best ranges from ~0 to ~10 for Ackley
    weight_ei = np.clip(f_best / 5.0, 0.3, 1.0)
    weight_pi = 1.0 - weight_ei
    
    # Combine: normalize EI by typical scale
    # EI has units of objective, PI is probability [0,1]
    # Scale EI to be comparable to PI
    ei_scaled = ei / (np.std(mu) + 1e-9)
    
    combined = weight_pi * pi + weight_ei * ei_scaled
    
    # Normalize by cost to the power 0.5
    scores = combined / np.sqrt(cost)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
