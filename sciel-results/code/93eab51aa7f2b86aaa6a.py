"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # 285s safety margin (limit is 300s)


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: prefer low cost and low mu
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Handle edge cases for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # EI can be negative if mu > f_best; clip to 0 for stability
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    score_ei = ei / cost
    
    # Add a UCB component to encourage exploration in uncertain regions
    # Use an adaptive beta based on the scale of sigma
    sigma_mean = np.mean(sigma)
    beta = 0.5 * sigma_mean
    
    ucb = -mu + beta * sigma
    # Normalize UCB to be comparable with EI
    score_ucb = ucb / cost
    
    # Combine EI and UCB
    # Weight EI more heavily since it's the primary acquisition function
    score = 0.85 * score_ei + 0.15 * score_ucb
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Add a tiny noise to break ties and ensure diversity
    rng = np.random.default_rng(42)
    score += rng.normal(0, 1e-8, size=score.shape)
    
    return score
# EVOLVE-BLOCK-END
