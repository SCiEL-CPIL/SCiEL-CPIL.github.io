"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a UCB variant with adaptive exploration.
    """
    t_start = time.time()
    DEADLINE = 285.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]

    # Sanitize
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Compute a progress-based exploration parameter
    # Early: high exploration, Late: high exploitation
    # Use the ratio of current best to a reference scale
    # For griewank 2d, values are in [0, 1] typically, with min near 0
    # f_best starts high and decreases
    # We can use f_best itself as a progress indicator: high f_best = early
    # But f_best is the current best observed value
    # Let's use the number of evaluations as a proxy, but we don't have it directly
    # Instead, use the scale of sigma relative to mu spread

    # Adaptive beta: starts high, decreases as we get more confident
    # Use the median sigma as a measure of remaining uncertainty
    sigma_med = np.median(sigma)
    mu_med = np.median(np.abs(mu))
    
    # If sigma is large relative to mu, we're still exploring
    # If sigma is small, we're exploiting
    ratio = sigma_med / (mu_med + 1e-8)
    
    # Map ratio to beta: high ratio -> high beta
    # Use a sigmoid-like mapping
    # beta ranges from ~0.5 to ~3.0
    beta = 0.5 + 2.5 * (1.0 / (1.0 + np.exp(-2.0 * (ratio - 1.0))))
    
    # Also scale down beta as f_best gets close to 0 (converging)
    # f_best for griewank: min is 0, so if f_best is small, we're close
    # Scale: if f_best < 0.01, reduce beta significantly
    f_scale = 1.0 / (1.0 + 50.0 * f_best)
    beta = beta * f_scale + 0.3 * (1.0 - f_scale)
    
    # UCB score
    ucb = mu + beta * sigma
    
    # Cost-aware: divide by cost
    score = -ucb / cost
    
    # Add a small exploration bonus for points with high uncertainty
    # relative to the overall uncertainty
    sigma_norm = sigma / (sigma_med + 1e-10)
    exploration_bonus = 0.1 * np.log1p(sigma_norm) / cost
    
    score = score + exploration_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline check
    if time.time() - t_start > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
