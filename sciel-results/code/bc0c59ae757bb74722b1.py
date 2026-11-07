"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of shekel 4d.
    
    Uses Expected Improvement per unit cost (standard cost-aware EI).
    """
    # Expected improvement per unit cost
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) / cost + sigma * norm.pdf(z) / cost
    # Ensure sigma is positive to avoid numerical issues
    return ei
# EVOLVE-BLOCK-END
