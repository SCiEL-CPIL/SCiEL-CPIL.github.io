"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma has a small floor to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: 
    # Use a slightly sublinear cost penalty (exponent 0.8) to be cost-aware
    # without being overly aggressive. Cost is in [1, 2], so cost^0.8 is in
    # [1, ~1.74], providing mild differentiation.
    cost_base = np.maximum(cost, 1e-6)
    cost_penalized = cost_base ** 0.8
    
    # Exploration bonus: combine normalized uncertainty with a UCB-like term
    # to encourage exploration in uncertain regions and help escape local basins.
    sigma_max = np.max(sigma_safe)
    ei_max = np.max(ei)
    
    if sigma_max > 1e-9 and ei_max > 1e-12:
        # Normalized uncertainty component (relative to max sigma)
        norm_uncertainty = sigma_safe / sigma_max
        # Scale by max EI to keep units consistent
        exploration_bonus = 0.15 * norm_uncertainty * ei_max
        # Add a small UCB-like term: beta * sigma / cost^0.8
        # This provides additional exploration pressure proportional to uncertainty
        beta_ucb = 0.05
        ucb_term = beta_ucb * sigma_safe / cost_penalized
    else:
        exploration_bonus = 0.0
        ucb_term = 0.0
    
    # Combined score: (EI + exploration) / cost^0.8 + UCB term
    scores = (ei + exploration_bonus) / cost_penalized + ucb_term
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
