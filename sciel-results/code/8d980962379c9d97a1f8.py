# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Log-Expected Improvement (Log-EI) for minimization.
    # Log-EI stabilizes the acquisition function by compressing the range of EI values.
    # This helps in balancing exploration and exploitation more effectively.
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate the normalized distance from the best found value
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI scores
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Calculate Log-EI scores
    # Add a small epsilon to avoid log(0)
    scores = np.log(ei + 1e-10)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
