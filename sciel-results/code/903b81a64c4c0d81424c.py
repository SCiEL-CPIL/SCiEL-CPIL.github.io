import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive PoI/EI mix with uncertainty bonus
    # PoI (Probability of Improvement) is more exploration-friendly than EI
    poi = norm.cdf(z)
    
    # Blend EI and PoI: use PoI for exploration, EI for exploitation
    # Weight based on how confident we are (higher sigma -> more exploration)
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    if sigma_max > 0:
        # When sigma is high relative to max, favor PoI (exploration)
        # When sigma is low, favor EI (exploitation)
        blend_weight = sigma / (sigma + 0.1 * sigma_max)
    else:
        blend_weight = np.zeros_like(sigma)
    
    # Combined improvement metric
    improvement = blend_weight * poi + (1 - blend_weight) * ei
    
    # Additional exploration bonus scaled by uncertainty
    exploration_bonus = 0.3 * sigma
    
    # Total acquisition score
    total_score = improvement + exploration_bonus
    
    # Normalize by cost
    score = total_score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
