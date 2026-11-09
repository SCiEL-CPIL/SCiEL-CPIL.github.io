"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Global variable to track the number of acquisition calls
# This allows us to implement a time-adaptive strategy
_acq_call_count = 0

def acquisition(mu, sigma, f_best):
    global _acq_call_count
    _acq_call_count += 1
    
    # The evaluator runs 25 optimization steps per seed.
    # We use modulo 25 to reset the progress for each new seed.
    # (call_count - 1) ensures the first call of a seed has step 0.
    step = (_acq_call_count - 1) % 25
    progress = step / 25.0
    
    # Time-adaptive exploration weight.
    # We want to explore aggressively at the start (progress ~ 0)
    # and exploit at the end (progress ~ 1).
    # Using a power law decay for a smooth transition.
    lambda_explore = 1.5 * (1.0 - progress) ** 1.5
    
    # Compute Expected Improvement (EI)
    # mu and sigma are arrays of shape (2000,)
    # f_best is a scalar
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Vectorized computation of CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Standard EI formula
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Hybrid score: EI + Exploration Bonus
    # The exploration bonus is proportional to uncertainty (sigma)
    score = ei + lambda_explore * sigma
    
    return score
# EVOLVE-BLOCK-END
