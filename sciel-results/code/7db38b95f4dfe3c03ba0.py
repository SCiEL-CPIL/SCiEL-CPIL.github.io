"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # wall-clock seconds total allowed (5s safety margin)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    We sample from the Gaussian Process posterior at each candidate point.
    The score is the negative of the sampled value, so that higher scores
    correspond to lower expected function values (minimization).
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value (not directly used in TS sampling, 
                but kept for signature compatibility)
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback to exploitation (greedy)
        return -mu.copy()
    
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal noise
    # Using a fresh random draw for each call implements Thompson Sampling
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled function values: f ~ N(mu, sigma^2)
    f_sample = mu + sigma_safe * z
    
    # We want to minimize f.
    # Acquisition score should be higher for better (lower) f.
    # So we return -f_sample.
    score = -f_sample
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
