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
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This function is superior to POI as it accounts for the magnitude of improvement,
    # helping to balance exploration and exploitation better in multimodal landscapes.
    improvement = (f_best - mu)
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    scores = improvement * cdf_z + sigma_safe * phi_z
    
    # EI is theoretically non-negative, but numerical precision might yield small negatives
    scores = np.maximum(scores, 0.0)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
