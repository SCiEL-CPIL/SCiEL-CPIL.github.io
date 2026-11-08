"""Cost-aware acquisition for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _safe_score(val, default=0.0):
    """Ensure score is finite and non-negative-ish."""
    if not np.isfinite(val):
        return default
    return val


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement per unit cost,
    with a small exploration bonus to avoid premature convergence.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed function value
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF/PDF arguments
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: EI per unit cost
    # cost is 1 + mean(x_scaled) where x_scaled in [0,1], so cost in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    
    score = ei / cost_safe
    
    # Add a small exploration bonus weighted by uncertainty to encourage
    # exploring uncertain regions, but keep it small so exploitation dominates
    # when EI is high. The bonus helps when EI is ~0 everywhere.
    # Use a temperature that decays with f_best (as we get better, explore less)
    # f_best starts around 0.5-1.0 for Griewank and decreases to ~0
    # We want exploration to be meaningful early but vanish as we converge
    temp = np.maximum(1.0 - f_best, 0.0) * 0.1
    
    # Exploration bonus: proportional to sigma, scaled by temperature
    # This is a soft UCB-like term
    exploration = temp * sigma_safe / cost_safe
    
    total_score = score + exploration
    
    # Ensure all values are finite
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    # Ensure non-negative (EI can be negative if mu > f_best, but we want
    # to still prefer lower-cost points in that case)
    # Actually, keep negative scores as they are - they just mean "don't pick"
    # But we must return finite values. Let's clip to a small negative floor
    # to avoid issues, but the relative ordering is what matters.
    
    return total_score
# EVOLVE-BLOCK-END
