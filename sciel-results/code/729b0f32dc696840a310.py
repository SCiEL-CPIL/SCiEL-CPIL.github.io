"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Expected Improvement (EI) with a small exploration bonus.
    # mu: shape (2000,) predicted mean
    # sigma: shape (2000,) predicted std dev
    # f_best: best observed value so far (lower is better)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term to encourage exploring high-uncertainty regions
    # This helps escape local optima in the acquisition function landscape
    # The exploration term is proportional to sigma, scaled down to not dominate
    exploration_bonus = 0.05 * sigma_safe
    
    # Combined score: higher is better
    score = ei + exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
