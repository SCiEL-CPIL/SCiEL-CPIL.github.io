"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return deterministic scores based on sigma to avoid randomness overhead
        # and ensure we return something valid quickly
        return sigma.copy()

    # Strategy: Expected Improvement (EI)
    # This balances exploration (high sigma) and exploitation (low mu).
    # For minimization, we define improvement as f_best - y.
    # EI = E[max(f_best - y, 0)]
    
    # Ensure sigma is positive and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma
    
    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF of the standard normal and phi is the PDF
    
    # Handle potential numerical issues with very large z
    # For very large positive z, Phi(z) -> 1, phi(z) -> 0. EI -> f_best - mu.
    # For very large negative z, Phi(z) -> 0, phi(z) -> 0. EI -> 0.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure scores are finite
    scores[~np.isfinite(scores)] = 0.0
    
    # EI is non-negative by definition, but due to floating point errors 
    # or edge cases, it might dip slightly below 0. 
    # We can clip to 0 to be safe, though argmax is unaffected by small negatives.
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
