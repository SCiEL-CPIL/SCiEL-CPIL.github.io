"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # 30 seconds margin from 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: Array of shape (2000,) with GP mean predictions.
        sigma: Array of shape (2000,) with GP standard deviation predictions.
        f_best: Scalar, the lowest observed function value.
        cost: Array of shape (2000,) with evaluation costs.
        
    Returns:
        Array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: Return scores based on mean (exploitation)
        # Minimizing mu / cost
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, -1e10)

    # Ensure sigma is positive and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.clip(sigma, 1e-9, 1e6)
    
    # Ensure cost is positive and finite
    cost = np.asarray(cost, dtype=np.float64)
    cost = np.clip(cost, 1e-9, 1e6)
    
    # Thompson Sampling:
    # Draw a sample from the posterior Gaussian distribution N(mu, sigma^2)
    # We draw a single standard normal sample and scale it.
    eps = np.random.normal(0.0, 1.0, size=mu.shape)
    f_sample = mu + sigma * eps
    
    # Objective: Minimize f(x) / cost(x)
    # Score: -f_sample / cost
    scores = -f_sample / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
