"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: EI + exploration bonus, normalized by cost
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward high uncertainty
    kappa = 0.5
    exploration = kappa * sigma_safe
    
    # Combined score, cost-normalized
    score = (ei + exploration) / np.maximum(cost, eps)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
