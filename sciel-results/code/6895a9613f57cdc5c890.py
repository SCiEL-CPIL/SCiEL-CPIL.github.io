# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Regret (ER) normalized by cost.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    sqrt2 = np.sqrt(2.0)
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    # Clip phi_z to prevent underflow issues in log-space or division, 
    # though here it's additive.
    phi_z = np.maximum(phi_z, 1e-15)
    
    cdf_z = 0.5 * (1.0 + erf(z / sqrt2))
    
    # Expected Regret (Equivalent to EI for minimization)
    er = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Ensure non-negative
    er = np.maximum(er, 0.0)
    
    # Cost-aware normalization
    # Cost is in [1, 2]. 
    # Using division by cost penalizes expensive points.
    cost_safe = np.maximum(cost, 1e-8)
    
    # Base score: Expected Regret per unit cost
    scores = er / cost_safe
    
    # Exploration bonus:
    # Encourage exploration in high-uncertainty regions.
    # Weighted by 1/cost to maintain cost-awareness.
    # Tuning parameter 0.05 (increased from 0.03) to better handle multimodality.
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
