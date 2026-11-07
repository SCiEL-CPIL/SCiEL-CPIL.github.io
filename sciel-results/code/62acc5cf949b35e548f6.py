"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balance expected improvement against evaluation cost.
    Uses EI per unit cost with a small exploration bonus for uncertain regions.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time to avoid timeout
    # The evaluator kills after 330s, so we add a safety margin
    t_start = time.time()
    
    # Convert inputs to arrays and ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Penalize high cost: normalize by mean cost (approx 1.5)
    # This makes the average cost have a multiplier of 1.0
    # Cheap points (cost=1) get 1.5x, expensive (cost=2) get 0.75x
    scores = ei * (1.5 / cost)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > 320:  # Safety margin before 330s limit
        # Return the current scores immediately
        return scores
    
    return scores
# EVOLVE-BLOCK-END
