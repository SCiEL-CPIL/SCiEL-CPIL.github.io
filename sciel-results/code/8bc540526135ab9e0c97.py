"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 295.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition per unit cost.
    
    mu: (2000,) predicted mean in objective units
    sigma: (2000,) predicted std in objective units
    f_best: float, lowest observed value
    cost: (2000,) evaluation cost per candidate
    
    Returns: (2000,) finite scores, higher = evaluate next
    """
    # Deadline guard: if we're running long, return a safe fallback
    if time.time() - _start_time > _DEADLINE_SECONDS:
        # Fallback: simple UCB to keep the loop moving
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        fallback = (f_best - mu + 0.5 * sigma_safe) / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure sigma is non-negative and numerically safe
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost is positive by problem definition, but guard against zeros
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample from the posterior of the objective at each candidate
    # For minimization, we want to find where the function value is low
    # Sample: f_sample = mu + sigma * z, where z ~ N(0,1)
    # We want to maximize the "improvement" score: f_best - f_sample
    # But we also need to account for cost
    
    # Generate random normal samples
    z = np.random.standard_normal(size=mu.shape)
    
    # Sample from the posterior
    f_sample = mu + sigma * z
    
    # The improvement over the best: f_best - f_sample
    # Higher improvement means better candidate
    improvement = f_best - f_sample
    
    # Normalize by cost to get cost-aware score
    ts_score = improvement / cost
    
    # Clip to ensure finite and reasonable range
    ts_score = np.clip(ts_score, -1e6, 1e6)
    
    # Replace any non-finite values
    ts_score = np.where(np.isfinite(ts_score), ts_score, 0.0)
    
    return ts_score
# EVOLVE-BLOCK-END
