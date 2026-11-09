"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling heuristic acquisition function.
    
    Scores candidates based on negative mean minus a dynamic, quantile-normalized
    standard deviation term. The exploration weight is scaled by the current best
    value to aggressively target high-variance regions early and sharply focus
    on the global minimum as f_best approaches zero.
    """
    # Compute quantiles of sigma for dynamic normalization
    # Use robust quantiles to handle outliers
    sigma_q05 = np.percentile(sigma, 5)
    sigma_q95 = np.percentile(sigma, 95)
    sigma_range = sigma_q95 - sigma_q05
    if sigma_range < 1e-12:
        sigma_range = 1e-12
    
    # Normalize sigma to [0, 1] range using quantiles
    sigma_norm = (sigma - sigma_q05) / sigma_range
    sigma_norm = np.clip(sigma_norm, 0, 1)
    
    # Dynamic exploration weight: 
    # When f_best is large (early), explore more aggressively
    # When f_best is small (late), focus on exploitation
    # Use a smooth transition based on f_best
    # Scale f_best to get exploration weight
    # Ackley 2D has minimum 0, typical values up to ~20
    # Use a sigmoid-like scaling
    f_best_scaled = np.abs(f_best)
    # Exploration weight: high when f_best is large, low when f_best is near 0
    # Use exponential decay: weight = 1 / (1 + f_best)
    # But we want aggressive early, sharp late
    # Let's use: weight = max(0.1, min(1.0, 1.0 / (1.0 + 0.5 * f_best_scaled)))
    exploration_weight = 1.0 / (1.0 + 0.5 * f_best_scaled)
    # Clamp to reasonable range
    exploration_weight = np.clip(exploration_weight, 0.05, 1.0)
    
    # Thompson Sampling score: negative mean - exploration_weight * normalized_sigma
    # Higher score means more promising (lower expected value with uncertainty bonus)
    score = -mu - exploration_weight * sigma_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
