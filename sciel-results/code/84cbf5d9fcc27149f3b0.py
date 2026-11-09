# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = None
DEADLINE_FRACTION = 0.92  # use 92% of 300s to be safe


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    We sample from the posterior at each candidate point and return the
    negative of the sample. The evaluator picks the maximum score, which
    corresponds to the minimum sampled function value.
    """
    global _start_time
    
    # Ensure timer is initialized
    if _start_time is None:
        _start_time = time.perf_counter()
    
    elapsed = _get_elapsed()
    
    # Deadline guard: if we're running out of time, fall back to simple EI
    # to ensure we return quickly
    if elapsed > DEADLINE_FRACTION * 300.0:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Sample from the posterior
    # Use a fixed random seed? No, the evaluator runs multiple seeds.
    # We should use the global random state or a local one?
    # Using np.random.rand is fine as it uses the global state, which is different per seed run.
    # However, to be safe and deterministic within a single acquisition call,
    # we can just generate random numbers.
    
    # Sample y ~ N(mu, sigma^2)
    # To avoid huge values, we can clip sigma or just use it as is.
    # Since we return -y, and y is in objective units, this is fine.
    
    # Generate random standard normal variables
    z_samples = np.random.randn(len(mu))
    
    # Sampled values
    sampled_values = mu + sigma * z_samples
    
    # We want to minimize, so we return -sampled_values
    # Higher score (less negative) means better (lower function value)
    scores = -sampled_values
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
