"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB-style exploration
    with adaptive kappa and cost normalization.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (stable computation)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB: mu + kappa * sigma
    # Use adaptive kappa based on the ratio of EI to sigma
    # When EI is large relative to sigma, we exploit (lower kappa)
    # When EI is small, we explore (higher kappa)
    ei_to_sigma = ei / (sigma_safe + 1e-10)
    # kappa in [0.3, 1.5], higher when less certain
    kappa = 0.3 + 1.2 * np.exp(-ei_to_sigma)
    
    ucb = mu + kappa * sigma_safe
    
    # Also compute a lower-bound estimate: for minimization,
    # we want to find points where mu is low and sigma is moderate
    # Use a hybrid: combine EI (exploit known good) with UCB (explore uncertain)
    
    # Normalize EI
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # Normalize UCB for minimization: lower UCB is better
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = 1.0 - (ucb - ucb_min) / ucb_range  # invert so higher = better
    
    # Add a pure exploration term: sigma normalized
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Blend: EI dominant, UCB for balanced exploration, sigma for pure exploration
    blended = 0.5 * ei_norm + 0.35 * ucb_norm + 0.15 * sigma_norm
    
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
