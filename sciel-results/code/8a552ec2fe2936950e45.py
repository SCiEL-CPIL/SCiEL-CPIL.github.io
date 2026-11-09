"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    with adaptive cost penalty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from the GP
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next)
    """
    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (exploration term)
    # Use a beta parameter that balances exploitation and exploration
    beta = 0.5
    lcb = mu - beta * sigma_safe
    
    # Cost-aware weighting
    # Normalize cost to [0, 1] for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Combine EI and LCB with cost penalty
    # Use exponential cost penalty to strongly favor low-cost regions
    cost_penalty = np.exp(-2.0 * cost_norm)
    
    # Weighted combination: EI for exploitation, LCB for exploration
    # Scale EI to be comparable with LCB
    ei_scale = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_scaled = ei / ei_scale
    
    # Final score: combine both terms with cost penalty
    # Higher score = better candidate
    scores = (0.6 * ei_scaled + 0.4 * (-lcb / (np.max(-lcb) if np.max(-lcb) > 0 else 1.0))) * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
