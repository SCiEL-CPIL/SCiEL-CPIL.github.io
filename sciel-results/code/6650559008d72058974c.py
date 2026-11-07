"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 285.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses Thompson Sampling: sample from the predictive posterior and
    select the candidate with the lowest sampled value (highest score = -sample).
    This naturally balances exploration and exploitation.
    """
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Thompson Sampling: draw a single sample from N(mu, sigma^2) for each candidate
    # Use a fixed seed based on the call to ensure reproducibility within a run
    # but different samples across calls (using a counter or time-based seed)
    # We use a deterministic approach: the sample is mu + sigma * z where z ~ N(0,1)
    # To get a single consistent sample per acquisition call, we generate one z-vector
    rng = np.random.default_rng()
    z = rng.standard_normal(n)
    
    # Sample from the predictive posterior
    samples = mu + sigma * z
    
    # Score: higher is better, so we negate the sample (lower objective = higher score)
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
