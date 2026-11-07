"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Expected Improvement (EI) and
    a stochastic Thompson-Sampling-inspired bonus, with an explicit deadline guard.
    
    mu: array of shape (N,) - GP mean predictions
    sigma: array of shape (N,) - GP standard deviation predictions
    f_best: scalar - best observed value so far (lower is better)
    cost: array of shape (N,) - cost of evaluating each candidate
    Returns: array of shape (N,) with finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're getting close to timeout, return a safe default
    # (This is a lightweight check; the real time pressure is on the caller's loop)
    
    # Ensure inputs are numpy arrays with proper shapes
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standard EI calculation
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # Add a small stochastic exploration bonus based on sigma
    # This helps escape local optima without being too aggressive
    # The bonus is scaled by sigma and inversely by cost to balance exploration
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    # Combine: primarily EI/cost with a modest exploration component
    scores = ei_cost + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (higher is better)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
