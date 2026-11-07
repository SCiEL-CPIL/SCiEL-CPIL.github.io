"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses Expected Improvement (EI) for minimization.
    EI balances exploration and exploitation by estimating the expected
    amount by which a candidate point could improve the current best.
    
    mu: (2000,) posterior mean in objective units
    sigma: (2000,) posterior standard deviation in objective units
    f_best: scalar, lowest observed value
    
    Returns: (2000,) scores, higher = more promising
    """
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive (potential for improvement)
    # If mu > f_best, z is negative (unlikely to improve)
    z = (f_best - mu) / sigma
    
    # Standard Normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is always non-negative.
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
