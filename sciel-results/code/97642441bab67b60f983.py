"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    # Deadline Guard: 300s limit, use 280s as safety margin
    if time.time() - _bo_start_time > 280:
        # If we are close to timeout, prioritize speed and exploitation
        return -mu

    # Thompson Sampling
    # Sample from the GP posterior and select the candidate with the lowest sample value.
    # This balances exploration and exploitation naturally.
    
    # Ensure sigma is positive and not too small to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate samples
    # Using global np.random for true stochasticity
    samples = np.random.normal(mu, sigma_safe)
    
    # Score is negative of the sample, so higher score means lower predicted value
    scores = -samples
    
    # Ensure finite values
    # If sigma was 0, samples == mu, so -samples == -mu
    # If any inf/nan, fallback to -mu
    mask = ~np.isfinite(scores)
    if np.any(mask):
        scores[mask] = -mu[mask]
        
    return scores
# EVOLVE-BLOCK-END
