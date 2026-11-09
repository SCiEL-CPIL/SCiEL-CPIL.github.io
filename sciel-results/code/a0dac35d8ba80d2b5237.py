import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Probability of Improvement (POI) weighted by a dynamic exploration
    factor, normalized by cost to prioritize high-probability improvements
    in low-cost regions.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: float, the best (lowest) observed value so far
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure sigma is non-negative and add a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement for minimization
    # POI = Phi(z) where z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Ensure POI is in [0, 1]
    poi = np.clip(poi, 0.0, 1.0)
    
    # Dynamic exploration factor based on global maximum uncertainty
    # This encourages exploration in early stages when uncertainty is high
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-8:
        # Normalize sigma by max to get relative uncertainty
        relative_uncertainty = sigma_safe / max_sigma
        # Exploration weight: higher when overall uncertainty is high
        # Use a sigmoid-like function to smoothly transition
        exploration_weight = 0.1 * (1.0 / (1.0 + np.exp(-5.0 * (max_sigma - 0.5))))
    else:
        exploration_weight = 0.0
        relative_uncertainty = np.zeros_like(sigma_safe)
    
    # Exploration bonus: weighted by relative uncertainty
    exploration_bonus = exploration_weight * relative_uncertainty
    
    # Cost normalization: divide by cost to get "improvement per unit cost"
    # Cost ranges from 1 to 2, so this gives a factor of 0.5-1.0
    cost_safe = np.maximum(cost, 1e-8)
    cost_normalized = 1.0 / cost_safe
    
    # Combine POI with exploration, then normalize by cost
    base_score = poi + exploration_bonus
    score = base_score * cost_normalized
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
