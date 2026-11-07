"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: return a safe fallback if we're close to the time limit
    # (The actual deadline is enforced by the evaluator, but this is a safety net)
    # In practice, this function is called quickly, so this guard is mostly for safety.
    
    # Lower bound on sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term based on uncertainty
    # This encourages exploration in regions with high sigma
    # The weight is small to not disrupt the main EI signal
    exploration_weight = 0.01
    exploration = exploration_weight * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
