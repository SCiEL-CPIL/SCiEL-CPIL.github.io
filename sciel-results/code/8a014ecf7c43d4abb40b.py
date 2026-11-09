"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function using Expected Improvement per unit cost."""
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (stable computation)
    # For large positive z, EI ≈ f_best - mu
    # For large negative z, EI ≈ 0
    ei = np.where(
        z > 0,
        (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z),
        0.0
    )
    
    # Cost-aware score
    return ei / cost
# EVOLVE-BLOCK-END
