"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Known minimum of Hartmann 6D
HARTMANN_6D_MIN = -3.32237

# Deadline guard: evaluator kills after 330s, use 325s as safety margin
_DEADLINE = 325.0
_start_time = None


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6D.
    
    Combines the structural idea of a smooth EI/UCB blend (from inspiration 2)
    with the simple, high-performing UCB-per-cost core (from inspiration 1),
    plus the sqrt(cost) normalization insight from the cell findings.
    
    The key insight: use a smooth blend between EI (exploitation) and UCB
    (exploration) where the blend weight is based on the relative uncertainty,
    then normalize by sqrt(cost) to balance exploration in higher-cost regions
    while still favoring cheaper evaluations.
    
    All inputs and output have shape (2000,).
    Returns 2000 finite scores (higher = evaluate next).
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety: return a reasonable default
        return np.ones_like(mu) * 1e-10
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement (in objective units, higher = better improvement)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term (in objective units, higher = more promising)
    # For minimization: f_best - mu + kappa * sigma
    kappa = 1.0
    ucb = f_best - mu + kappa * sigma
    
    # Smooth blend between EI and UCB
    # When sigma is large relative to |f_best - mu|, favor UCB (explore)
    # When sigma is small, favor EI (exploit)
    # Use a smooth sigmoid-like blend
    abs_diff = np.abs(f_best - mu)
    blend_raw = sigma / (abs_diff + sigma + 1e-9)
    # blend_raw is in [0, 1], higher when uncertainty dominates
    # Scale to moderate blend: max 0.5 weight on UCB
    blend = 0.5 * blend_raw
    
    # Combine: (1 - blend) * EI + blend * UCB
    combined = (1.0 - blend) * ei + blend * ucb
    
    # Normalize by sqrt(cost) to balance exploration vs cost
    # This allows higher-cost regions to be considered if they promise enough improvement
    # while still favoring cheaper evaluations
    sqrt_cost = np.sqrt(cost)
    scores = combined / sqrt_cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
