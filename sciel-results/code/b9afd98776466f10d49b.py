"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by cost.
    
    EI directly estimates the expected reduction in the objective value.
    By dividing by cost, we prioritize points that offer the best improvement
    per unit of evaluation cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure positive sigma and cost for division and log calculations
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)

    # Compute Z score: (f_best - mu) / sigma
    # This represents how many standard deviations f_best is above the mean
    z = (f_best - mu) / sigma_safe

    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # Calculate Phi(z) and phi(z)
    # Using scipy.stats.norm
    phi_z = norm.pdf(z)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    cdf_z = norm.cdf(z)
    cdf_z = np.where(np.isfinite(cdf_z), cdf_z, 0.0)

    # Calculate EI
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical errors might cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    score = ei / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
