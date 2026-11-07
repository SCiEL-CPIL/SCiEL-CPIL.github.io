"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Thresholded Expected Improvement (cEI) acquisition function.
    
    Strategy:
    Uses Expected Improvement (EI) but subtracts a linear cost penalty.
    This models the cost as a direct utility loss. Unlike EI/cost, which 
    compresses high-EI values, this preserves the magnitude of improvement 
    potential while explicitly discouraging expensive evaluations unless 
    the expected improvement is sufficiently large.
    
    Formula: Score = EI - lambda * cost
    We use lambda = 0.1 based on the observation that small penalties 
    work well for this problem class.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple greedy exploitation if time is critical
        # Pick the point with the lowest mu, penalized by cost
        scores = -mu - 0.1 * cost
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in CDF/PDF
    z_clipped = np.clip(z, -30, 30)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical artifacts might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Subtract cost penalty
    # This creates a threshold: EI must be > lambda * cost to be positive
    lambda_cost = 0.1
    scores = ei - lambda_cost * cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
