"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with
    a time-budget guard.
    
    mu, sigma: arrays of shape (2000,) - GP posterior mean and std
    f_best: scalar - best observed value so far
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns: array of shape (2000,) with finite scores (higher = evaluate next)
    """
    start_time = time.time()
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from the GP posterior
    # For each candidate, sample f(x) ~ N(mu, sigma^2)
    # We use a single sample to maintain consistency within one call
    rng = np.random.default_rng(seed=42)
    
    # Sample function values
    z = rng.standard_normal(n)
    sampled_values = mu - sigma_safe * z  # Note: lower is better, so we sample and minimize
    
    # Cost-aware score: we want to minimize sampled_value / cost
    # But acquisition returns higher = better, so we return negative of that
    # Actually, for Thompson Sampling in minimization, we select the point
    # that minimizes sampled_value / cost
    
    # The score should be such that higher = better
    # We want to pick the point with lowest sampled_value / cost
    # So score = - (sampled_value / cost)
    
    # Handle potential division issues
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute the ratio
    ratio = sampled_values / cost_safe
    
    # Higher score means we should evaluate this point
    # Since we want to minimize ratio, higher acquisition = lower ratio
    # So acquisition = -ratio
    scores = -ratio
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time budget check: if we're running low on time, return a simple fallback
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        # Fall back to a simpler cost-normalized improvement
        z_ei = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
