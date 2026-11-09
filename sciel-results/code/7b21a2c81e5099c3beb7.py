import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None
_seed_counter = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _seed_counter
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check if we're approaching the deadline (330s limit, use 325s as safety margin)
    elapsed = time.time() - _start_time
    if elapsed > 325:
        safe_cost = np.maximum(cost, 1e-10)
        return -mu / safe_cost
    
    # Small epsilon to prevent division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Thompson Sampling: sample a single latent function value from the predictive distribution
    # y ~ N(mu, sigma^2)
    # Use a deterministic seed based on the call count to ensure reproducibility within a run
    # but variation across calls (which is the point of TS)
    _seed_counter += 1
    rng = np.random.RandomState(_seed_counter % (2**31 - 1))
    
    # Sample from N(mu, sigma^2)
    # We want to minimize, so we score by -y / cost
    # To make it more stable, we can sample multiple times and take a quantile, or just one sample
    # Single sample is the standard Thompson Sampling approach
    samples = rng.normal(loc=mu, scale=safe_sigma)
    
    # Score is -sampled_value / cost
    # This naturally balances exploration (high sigma -> wider spread -> chance of very low sample)
    # and exploitation (low mu -> lower samples on average)
    score = -samples / safe_cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
