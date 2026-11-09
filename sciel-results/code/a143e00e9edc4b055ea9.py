"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the standardized improvement z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Use Expected Improvement (EI) instead of POI for better balance
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This rewards both the probability of improvement and the magnitude
    phi_z = norm.pdf(z)
    phi_z = np.nan_to_num(phi_z, nan=0.0, posinf=0.0, neginf=0.0)
    
    scores = (f_best - mu) * norm.cdf(z) + sigma_safe * phi_z
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
