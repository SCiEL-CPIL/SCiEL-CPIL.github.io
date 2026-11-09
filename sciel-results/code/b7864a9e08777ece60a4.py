import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return pure PMM scores
        scores = -np.asarray(mu, dtype=np.float64)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    
    # Posterior Mean Minimization (PMM):
    # Score = -mu, higher score means lower predicted mean
    # This ignores uncertainty (sigma) entirely, forcing
    # deterministic exploitation of the GP's predictive mean
    scores = -mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
