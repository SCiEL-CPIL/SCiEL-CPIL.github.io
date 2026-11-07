"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    n = len(mu)
    
    # Expected Improvement (EI)
    # EI(x) = E[max(f_best - f(x), 0)]
    # Given f(x) ~ N(mu, sigma^2)
    # Let z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Calculate z-scores
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z)
    # norm.cdf(z) gives Phi(z)
    # norm.pdf(z) gives phi(z)
    
    # To avoid numerical issues with very large/small z, we rely on scipy's robust implementation.
    # However, we can optimize by calculating only where necessary or using stable forms if needed.
    # For 2000 points, direct calculation is fast.
    
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Calculate EI
    # Term 1: (f_best - mu) * Phi(z)
    # Term 2: sigma * phi(z)
    ei_scores = (f_best - mu) * cdf_z + sigma * phi_z
    
    # EI should be non-negative. Due to floating point errors, small negative values might appear.
    # Clip to 0 to ensure higher scores are always better and no negative scores confuse the selection.
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Ensure all scores are finite
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
