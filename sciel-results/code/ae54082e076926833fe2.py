"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and LCB with cost normalization.
    
    Args:
        mu: GP mean predictions (2000,)
        sigma: GP standard deviation predictions (2000,)
        f_best: best observed value (scalar)
        cost: cost of evaluating each candidate (2000,)
    
    Returns:
        scores: (2000,) acquisition scores, higher = more desirable
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # GP-LCB (Lower Confidence Bound for minimization)
    # mu - beta * sigma, we negate to get higher = better
    beta = 2.0
    lcb = -(mu - beta * sigma_safe)
    
    # Combine EI and LCB with cost normalization
    # EI/cost favors efficient improvements
    # LCB/cost favors exploring uncertain regions efficiently
    
    ei_cost = ei / cost_safe
    lcb_cost = lcb / cost_safe
    
    # Blend: weight EI more when we have good exploitation signal, LCB for exploration
    # Use a simple weighted average
    scores = 0.6 * ei_cost + 0.4 * lcb_cost
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
