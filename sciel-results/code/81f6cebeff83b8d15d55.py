"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI)
    # Structural change: Replace LCB with EI/cost.
    # EI is non-linear and often better calibrated for optimization.
    # Dividing by cost directly optimizes for improvement per unit cost.
    
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    # Using scipy.stats for accuracy
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware score: EI / cost
    # Higher score means better expected improvement per unit cost
    score = ei / (cost + 1e-9)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
