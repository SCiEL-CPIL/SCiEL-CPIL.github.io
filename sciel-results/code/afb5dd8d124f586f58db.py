import numpy as np
import time

# Global state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Safety: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline Guard: 
    # If we have exceeded a safety margin (e.g., 300s out of 330s),
    # fall back to a very fast deterministic strategy to ensure return.
    # In this case, pure exploitation (minimize mu).
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Return scores such that the evaluator picks the lowest mu.
        # Evaluator picks max score. So we return -mu.
        return -mu
        
    # Thompson Sampling
    # Draw a sample from the posterior distribution for each candidate
    # y ~ N(mu, sigma)
    # We want to minimize y, so we maximize -y.
    z = np.random.randn(mu.shape)
    sample = mu + sigma * z
    scores = -sample
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
