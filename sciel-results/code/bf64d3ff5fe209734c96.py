import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for minimization.
    Draws a single sample from the posterior at each candidate point
    and returns the negative sample as the score (higher = better).
    """
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > 295:
        # Fallback: return scores that favor points with low mu
        mu_safe = np.asarray(mu, dtype=np.float64)
        mu_safe = np.where(np.isfinite(mu_safe), mu_safe, 0.0)
        return -mu_safe
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize sigma to avoid division by zero and handle non-finite values
    sigma = np.where(sigma > 1e-10, sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Thompson Sampling: draw a single sample from N(mu, sigma^2) at each point
    # Use a fixed random seed derived from the current state for reproducibility
    # within a single call, but different across calls
    rng = np.random.default_rng()
    
    # Draw samples: sample_i ~ N(mu_i, sigma_i^2)
    samples = mu + sigma * rng.standard_normal(size=mu.shape)
    
    # For minimization, we want to select the point with the lowest sample
    # So we return negative samples as scores (higher score = lower sample = better)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
