"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time and return early if approaching limit
_START_TIME = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    Uses posterior sampling to estimate Expected Improvement by drawing
    samples from the GP posterior and computing the expected maximum improvement.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: return current best if approaching 330s limit
    elapsed = time.time() - _START_TIME
    if elapsed > 325.0:  # Safety margin of 5s
        # Return a simple score that favors points with low mu
        return -mu
    
    n_samples = 50
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples from the GP posterior
    # For each candidate, we sample from N(mu, sigma^2)
    rng = np.random.default_rng(42)
    
    # Generate all samples at once for efficiency
    # Shape: (n_samples, 2000)
    samples = mu[None, :] + sigma_safe[None, :] * rng.standard_normal((n_samples, len(mu)))
    
    # For each sample, compute the improvement over f_best
    # Improvement is max(0, f_best - sample_value)
    improvements = np.maximum(0, f_best - samples)
    
    # Expected improvement is the mean of improvements across samples
    scores = np.mean(improvements, axis=0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
