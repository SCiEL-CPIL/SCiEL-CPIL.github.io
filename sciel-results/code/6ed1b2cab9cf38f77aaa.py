import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Thompson Sampling: sample from the posterior N(mu, sigma^2) and
    return the samples as scores (higher is better for selection, but since
    we want to minimize, we negate the samples).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0
    
    # Thompson Sampling: draw a sample from N(mu, sigma^2) for each candidate
    # Since we want to minimize, we negate the sample so that lower function
    # values (better) correspond to higher scores
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Negate for minimization (lower function value = higher score)
    scores = -samples
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
