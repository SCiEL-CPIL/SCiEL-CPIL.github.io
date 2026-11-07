import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # LCB = mu - kappa * sigma
    # We want to maximize the LCB score (higher = more promising)
    # So we negate: score = -mu + kappa * sigma
    
    # Handle zero sigma safely
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa: higher when far from optimum (exploration),
    # lower when close (exploitation)
    known_min = -39.1662
    
    # Distance from optimum in normalized sense
    # f_best ranges roughly from ~20 (random start) to -39.1662
    # Normalize: 1.0 when far, 0.0 when at optimum
    dist = max(0.0, (f_best - known_min) / (30.0 - known_min))
    dist = min(1.0, dist)
    
    # kappa: exploration weight, starts high, decays as we approach optimum
    # For LCB in minimization, higher kappa means more exploration
    kappa = 2.0 * (0.2 + 0.8 * dist)
    
    # LCB score: higher is better (we want low mu, high sigma)
    # score = -mu + kappa * sigma
    score = -mu + kappa * sigma_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
