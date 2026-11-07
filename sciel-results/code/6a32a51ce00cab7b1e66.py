"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with a small exploration bonus based on uncertainty.
    """
    # Avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Add a small uncertainty bonus to encourage exploration
    # This helps escape local optima in Michalewicz function
    exploration_bonus = 0.1 * sigma_safe
    
    # Final score: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
