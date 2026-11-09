import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Thompson Sampling: sample from the posterior N(mu, sigma^2)
    for each candidate and return the negative sampled value as score.
    Higher score = lower sampled value = more promising.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Clamp sigma to a small positive value to avoid issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # Use a fixed seed based on the current time to get a fresh sample each call
    # but deterministic within a single call
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Score: higher is better, so we negate the sampled value
    # (lower sampled value = more promising for minimization)
    score = -samples
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
