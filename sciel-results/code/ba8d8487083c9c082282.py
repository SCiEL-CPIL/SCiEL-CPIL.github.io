"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we don't exceed 300s wall-clock per evaluation
    # We track time at module level to avoid repeated overhead
    # Use a simple time check - if we're close to deadline, return quickly
    # But since this is called once per acquisition step, the main risk is
    # the computation itself. Let's optimize the computation.
    
    # Cost-aware acquisition: balance expected improvement against cost
    # Use a more robust formulation with proper handling of edge cases
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative EI
    ei = np.maximum(ei, 0)
    
    # Cost-aware: divide by cost, but add small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Return EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
