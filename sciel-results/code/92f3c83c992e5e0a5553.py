"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Calculate standard Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle division by zero or very small sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Variance-Weighting:
    # Multiply EI by normalized variance sigma^2 / max(sigma^2)
    # This boosts exploration in high-variance regions more aggressively than standard EI.
    
    max_sigma_sq = np.max(sigma ** 2)
    if max_sigma_sq < eps:
        # If variance is negligible everywhere, fall back to EI (or just EI)
        # Actually if max is 0, all are 0, so EI is 0.
        # But to be safe, if max is tiny, don't divide by tiny.
        weights = np.ones_like(sigma)
    else:
        weights = (sigma ** 2) / max_sigma_sq
        
    vw_ei = ei * weights
    
    # Ensure finite output
    vw_ei = np.nan_to_num(vw_ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return vw_ei
# EVOLVE-BLOCK-END
