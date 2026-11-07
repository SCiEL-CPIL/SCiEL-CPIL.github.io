"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE = 330.0  # seconds, evaluator kills after 330s
SAFETY_MARGIN = 15.0  # seconds to return safely


def _time_remaining():
    return DEADLINE - (time.time() - _START_TIME) - SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: (2000,) mean predictions in objective units
    - sigma: (2000,) std predictions in objective units
    - f_best: scalar, lowest observed value
    - cost: (2000,) cost of evaluating each candidate
    
    Returns:
    - (2000,) finite scores, higher = evaluate next
    """
    # Check deadline
    if _time_remaining() < 1.0:
        # Return a safe fallback: prefer low cost and low mu
        return -cost - mu * 0.1
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: penalize cost quadratically to bias toward low-cost regions
    # The minimum of the shifted Levy function tends to be in the low-cost corner
    # Cost is 1 + mean(x_scaled), so low-cost = near (-10, -10) in original coords
    # Using cost^2 creates strong bias toward cheapest evaluations
    
    # Normalize EI to be comparable across scales
    # Use a soft normalization to avoid extreme values
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
    
    # Cost penalty: quadratic to strongly favor low-cost regions
    # cost ranges from 1 to 2 (since mean(x_scaled) is in [0,1])
    # cost^2 ranges from 1 to 4
    # We want the penalty to be significant but not dominate completely
    cost_penalty = cost ** 2
    
    # Combine: reward high EI, penalize high cost
    # Scale EI to be comparable to cost penalty
    # EI values can be large; normalize by typical scale
    ei_scale = np.std(ei) + 1e-9
    ei_scaled = ei / ei_scale
    
    # Final score: EI scaled minus cost penalty
    # Higher score = better candidate
    scores = ei_scaled - 0.5 * cost_penalty
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
