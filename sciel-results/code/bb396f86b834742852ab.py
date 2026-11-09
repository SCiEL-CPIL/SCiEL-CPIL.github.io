import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and not too small
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in pdf/cdf for extreme values
    # norm.pdf(z) underflows for z < -37, norm.cdf(z) overflows for z > 37
    # Clipping to [-10, 10] is safe and fast
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Add a small exploration bonus to encourage diversity
    # This helps if the EI landscape is very flat
    exploration_bonus = 0.05 * sigma
    
    return ei + exploration_bonus
