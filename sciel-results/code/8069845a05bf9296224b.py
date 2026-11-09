"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Posterior Mean Minimization (PMM)
    # Directly minimize the GP predictive mean
    # Score = -mu (higher means lower predicted value, so we pick the best)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, np.inf)
    
    # PMM score: higher is better, so we negate the mean
    pmm_scores = -mu_safe
    
    # Ensure finite scores
    pmm_scores = np.where(np.isfinite(pmm_scores), pmm_scores, -np.inf)
    
    return pmm_scores
# EVOLVE-BLOCK-END
