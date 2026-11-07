"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Samples a function realization from the GP posterior and selects the point
    that minimizes the sampled value per unit cost.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far (unused in TS directly, but part of interface)
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        # Return a simple score based on cost only (cheaper is better)
        # Using a deterministic fallback to avoid random state issues near deadline
        return 1.0 / (cost + 1e-9)
    
    # Thompson Sampling:
    # 1. Sample a standard normal random variable for each candidate
    z = np.random.normal(size=mu.shape)
    
    # 2. Compute the sampled function values
    # f_sample(x) ~ N(mu(x), sigma(x)^2)
    f_sample = mu + sigma * z
    
    # 3. Compute the score: minimize f_sample / cost
    # We want to maximize the score, so we use negative of the ratio
    # score(x) = - f_sample(x) / cost(x)
    # This naturally balances exploration (via sigma) and exploitation (via mu)
    # and incorporates cost efficiency.
    
    # Ensure cost is positive to avoid division by zero or negative scores
    # Cost is defined as 1 + mean(x_scaled), so it is always >= 1.
    # But for safety:
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = -f_sample / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
