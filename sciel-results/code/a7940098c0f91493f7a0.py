"""Cost-aware acquisition using Expected Improvement with adaptive exploration for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) with
    adaptive exploration bonus for Griewank 2D Bayesian Optimization.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Compute EI
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (no improvement expected)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization
    # Use a soft cost normalization that penalizes expensive points
    # but doesn't completely eliminate them
    cost_penalty = 1.0 / (1.0 + cost_safe)
    
    # Adaptive exploration: add a small variance-based exploration bonus
    # This encourages exploration in uncertain regions
    # Normalize sigma to [0, 1] range for stable scaling
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_min = np.min(sigma_safe) if np.min(sigma_safe) < np.max(sigma_safe) else 0.0
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Exploration bonus: small weight on normalized uncertainty
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_norm * np.max(ei) if np.max(ei) > 0 else 0.0
    
    # Combine EI and exploration bonus
    score = ei + exploration_bonus
    
    # Apply cost penalty
    score = score * cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
