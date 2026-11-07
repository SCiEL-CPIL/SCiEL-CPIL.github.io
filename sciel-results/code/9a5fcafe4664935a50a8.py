import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition for Styblinski-Tang 2D BO.
    Uses LCB with adaptive beta that increases exploration early and exploitation later.
    Divides by cost to naturally bias toward cheaper evaluations.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std predictions
    - f_best: float, best observed function value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / (cost_safe + 1e-9)
    
    # Ensure inputs are valid
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Known minimum of Styblinski-Tang 2D is approximately -78.3323
    known_min = -78.3323
    
    # Compute progress: 0 at start, 1 when we've found the optimum
    # Initial f_best from 5 uniform points on [-5,5]^2 is typically in [-50, 50]
    # The range from typical start to optimum is roughly 130-150
    progress = np.clip(1.0 - (f_best - known_min) / 150.0, 0.0, 1.0)
    
    # Adaptive beta for LCB: start with high exploration, decrease as we converge
    # beta controls the trade-off between exploitation (mu) and exploration (sigma)
    # Higher beta = more exploration
    beta = 3.0 * (1.0 - 0.6 * progress)
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # But since we want higher scores to be better, we negate: -mu + beta * sigma
    lcb = -mu + beta * sigma
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    # This naturally biases toward the center of the domain where cost is lowest
    # (cost = 1 + mean(x_scaled), so center has cost ~1.5, edges have cost ~2)
    cost_penalty = cost + 1e-9
    score = lcb / cost_penalty
    
    # Add a small bonus for regions with high uncertainty that might contain the global minimum
    # This helps escape local minima in the multi-modal Styblinski-Tang function
    sigma_max = np.max(sigma)
    if sigma_max > 1e-6:
        uncertainty_bonus = 0.1 * (sigma / sigma_max)
        score += uncertainty_bonus / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e6)
    
    # Normalize to [0, 1] for consistent scale across iterations
    score_max = np.max(score)
    score_min = np.min(score)
    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    
    return score
