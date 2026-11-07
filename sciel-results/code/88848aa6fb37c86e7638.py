# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

# Precompute constants for speed
# The cost function is 1 + mean(x_scaled) where x_scaled = (x - (-4)) / 9
# We don't have x, but we have 'cost' which is 1 + mean(x_scaled)
# So mean(x_scaled) = cost - 1
# This doesn't directly help for the acquisition function unless we want to weight by cost.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) with predicted means
    - sigma: array of shape (2000,) with predicted standard deviations
    - f_best: float, the lowest observed value
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple fallback to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: Use a softer cost penalty to encourage exploration
    # Instead of EI / cost, use EI / (cost ** 0.5) to be less aggressive with cost
    # This allows more exploration in high-cost regions if they offer good EI
    scores = ei / np.sqrt(cost)
    
    # Add a small exploration bonus based on uncertainty
    # This helps avoid getting stuck in local optima
    exploration_bonus = sigma * 0.1
    
    scores = scores + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
