"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays for efficiency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - beta * sigma
    # So we maximize -(mu - beta * sigma) = -mu + beta * sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # 3. Normalize scores to combine them effectively
    # This makes the hybrid robust to scale differences between EI and LCB
    
    # Normalize EI
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        z_ei = np.zeros_like(ei)
    else:
        z_ei = (ei - ei_mean) / ei_std
        
    # Normalize LCB Score
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std < 1e-9:
        z_lcb = np.zeros_like(lcb_score)
    else:
        z_lcb = (lcb_score - lcb_mean) / lcb_std
        
    # 4. Combine with weights
    # Knowledge suggests 0.7 exploration (LCB) and 0.3 exploitation (EI) works well
    w_explore = 0.7
    w_exploit = 0.3
    
    scores = w_explore * z_lcb + w_exploit * z_ei
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
