"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # 10 seconds safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Thompson Sampling: draws a sample from the posterior N(mu, sigma)
    for each candidate and returns the negative sample as the score.
    Higher score means lower sampled value (better for minimization).
    
    mu, sigma: arrays of shape (2000,) with posterior mean and std dev
    f_best: scalar, the lowest observed value so far (not used in TS directly, 
            but kept for interface compatibility)
    
    Returns: array of 2000 finite scores (higher = more promising)
    """
    # Deadline guard: if we've been running too long, return a simple EI
    # to ensure we finish within the wall-clock limit.
    # However, TS is very fast (O(N)), so this is unlikely to trigger unless
    # the overhead of the Python call itself is high, which it isn't for N=2000.
    # We keep it as a structural safety net.
    if time.time() - _start_time > _DEADLINE:
        # Fallback to simple EI if we are somehow close to timeout
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        # Standard normal CDF approximation or use scipy if available, 
        # but numpy doesn't have norm.cdf. We'll stick to a simple heuristic fallback
        # that prioritizes low mu and high sigma to ensure finite output.
        # Actually, let's just return -mu + sigma to be safe and fast.
        return -(mu) + sigma

    # Thompson Sampling
    # Draw a sample from N(mu, sigma) for each candidate
    # We use a single random normal vector to generate all samples efficiently
    # z ~ N(0, 1)
    # sample = mu + sigma * z
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate random normal numbers
    # Using np.random.normal is fast for vectorized operations
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior
    samples = mu + sigma_safe * z
    
    # We want to minimize the objective, so we want the sample with the lowest value.
    # The acquisition score should be higher for better candidates.
    # So, score = -sample
    scores = -samples
    
    # Ensure all scores are finite
    # If sigma was NaN or Inf, samples might be invalid.
    # sigma_safe handles 0, but not NaN.
    # mu might be NaN? The problem states mu and sigma are from a GP posterior, 
    # which should be finite, but let's be defensive.
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
