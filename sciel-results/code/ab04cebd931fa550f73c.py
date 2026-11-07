"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost with numerical stability.
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute expected improvement using stable formulation
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For numerical stability, handle cases where z is very large or very small
    ei = np.zeros_like(mu)
    
    # Where z is positive (mu < f_best), EI is positive
    # Where z is negative (mu > f_best), EI can be negative but we clamp to 0
    # Actually EI should always be >= 0 for proper EI, but with this formulation
    # it can be negative when mu > f_best. Let's use the standard formulation.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI should be non-negative; clamp small negative values due to numerical issues
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
