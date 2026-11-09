"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) per unit cost with a small adaptive
    exploration bonus, all normalized by cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration bonus: use mean sigma as proxy for GP maturity
    # When mean sigma is large, GP is uncertain -> explore more
    # When mean sigma is small, GP is confident -> exploit more
    mean_sigma = np.mean(sigma_safe)
    # Map mean_sigma to a small bonus weight in [0.05, 0.15]
    bonus_weight = 0.05 + 0.10 * np.clip(mean_sigma, 0, 1)
    
    # Add exploration bonus
    exploration_bonus = bonus_weight * sigma_safe
    
    # Combine EI and exploration bonus
    combined_score = ei + exploration_bonus
    
    # Normalize by cost (cost is at least 1.0, so no division by zero)
    scores = combined_score / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
