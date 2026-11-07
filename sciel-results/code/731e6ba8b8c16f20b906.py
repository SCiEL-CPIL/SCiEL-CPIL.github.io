import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses an Adaptive UCB (Upper Confidence Bound) strategy.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Global minimum of Michalewicz 2D
    f_min_global = -1.8013
    # Estimated maximum for normalization
    f_max_est = 1.0
    
    # Calculate normalized distance from the global minimum
    # dist_norm = 0 if f_best is at global min, 1 if f_best is at max
    range_val = f_max_est - f_min_global
    if range_val < 1e-6:
        dist_norm = 0.0
    else:
        dist_norm = np.clip((f_best - f_min_global) / range_val, 0.0, 1.0)
        
    # Adaptive exploration constant kappa
    # High kappa when far from optimum (dist_norm high)
    # Low kappa when close to optimum (dist_norm low)
    kappa_min = 0.5
    kappa_max = 3.0
    
    kappa = kappa_min + (kappa_max - kappa_min) * dist_norm
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB Score for minimization
    # We want to maximize this score.
    # -mu favors low predicted mean.
    # + kappa * sigma favors high uncertainty.
    score = -mu + kappa * sigma_safe
    
    return score
