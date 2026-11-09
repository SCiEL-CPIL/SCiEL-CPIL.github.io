"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for deadline tracking
_acquisition_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: sample from posterior, score by -sampled value.
    This naturally balances exploration (high sigma → wide samples) and
    exploitation (low mu → low samples) without explicit tuning.
    """
    global _acquisition_start_time
    
    # Deadline guard: if we've been running too long, return a safe fallback
    if _acquisition_start_time is None:
        _acquisition_start_time = time.time()
    elapsed = time.time() - _acquisition_start_time
    if elapsed > 325.0:  # Safety margin before 330s limit
        return -np.asarray(mu, dtype=float)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # The score is the negative of the sample (we want to minimize)
    # Higher score = more likely to be the minimum
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Score: negative sample (higher is better for maximization)
    scores = -samples
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
