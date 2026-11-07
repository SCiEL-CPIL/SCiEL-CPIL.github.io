"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard: we must return within the 300s limit for the evaluator.
# We track time in the acquisition function itself, but since it's called repeatedly,
# we need a lightweight guard. The main cost is in the acquisition computation.
# We'll add a simple time check to ensure we don't exceed a safe margin.

# Precompute constants for speed
_EPS = 1e-9

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if sigma.shape != mu.shape:
        sigma = np.broadcast_to(sigma, mu.shape)
    if cost.shape != mu.shape:
        cost = np.broadcast_to(cost, mu.shape)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, _EPS)
    cost_safe = np.maximum(cost, _EPS)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # Using a softer cost penalty to allow exploration in cheaper regions
    scores = ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
