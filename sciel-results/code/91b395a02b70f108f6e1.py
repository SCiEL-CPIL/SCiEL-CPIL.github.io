"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement (EI)
    and Probability of Improvement (PI), normalized before cost division.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Track elapsed time for safety margin
    t_start = time.time()
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (stable computation)
    # For large positive z, EI ≈ f_best - mu
    # For large negative z, EI ≈ 0
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize EI and PI to [0, 1] range for blending
    # Use robust normalization: divide by max to avoid issues with very small values
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 0 else 1.0
    
    ei_norm = ei / ei_max
    pi_norm = pi / pi_max
    
    # Add a small uncertainty bonus to encourage exploration
    # Normalize sigma to [0, 1] range
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Blend EI, PI, and uncertainty with weights
    # EI is dominant, PI provides probability signal, uncertainty encourages exploration
    blended = 0.7 * ei_norm + 0.2 * pi_norm + 0.1 * sigma_norm
    
    # Cost-aware: divide by cost (higher cost = lower score)
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're getting close to the limit, return immediately
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
