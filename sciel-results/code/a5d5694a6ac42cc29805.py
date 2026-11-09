"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
# Set a deadline slightly before the hard kill to allow for safe return
# 330s is the kill time. We use 325s to be safe.
_DEADLINE = _start_time + 325.0 

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    # Ensure inputs are numpy arrays for consistent operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Deadline guard: If we are close to the limit, return a fast, conservative score
    # to ensure we finish safely. -mu/cost prioritizes exploitation of the current
    # best region while respecting cost.
    if time.time() > _DEADLINE - 2.0:
        scores = -mu / np.where(cost > 1e-9, cost, 1.0)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    
    # Compute Expected Improvement (EI) for minimization.
    # Improvement is defined as f_best - f(x).
    # E[Improve] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Add small epsilon to sigma to avoid division by zero
    sigma_safe = sigma + 1e-9
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # norm.cdf(-10) is ~7e-24, norm.cdf(10) is ~1.0
    # norm.pdf(10) is ~7.7e-23
    # This prevents underflow/overflow issues in the standard normal functions
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Cost-awareness: Divide by cost to prioritize cheaper evaluations.
    # The task states: "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    # Lower cost is better, so dividing the score (which represents potential gain)
    # by cost effectively maximizes gain per unit cost.
    scores = ei / cost_safe
    
    # Ensure all scores are finite. If not, replace with 0.0 (neutral/invalid).
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
