"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Compute Expected Improvement (EI) with a small exploration bonus.
    # EI is the standard acquisition for GP-based Bayesian optimization.
    # We add a small term that slightly favors regions with higher uncertainty
    # to avoid premature convergence, but keep it small so exploitation dominates.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small exploration bonus proportional to sigma
    # This helps escape local optima in the 1D sphere function
    exploration_weight = 0.1
    bonus = exploration_weight * sigma_safe
    
    scores = ei + bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
