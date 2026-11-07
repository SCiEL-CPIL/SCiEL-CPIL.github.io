"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0
_step_count = 0

def acquisition(mu, sigma, f_best):
    global _step_count
    _step_count += 1
    
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    # Thompson Sampling: sample from the predictive posterior
    # For each candidate, draw a sample from N(mu_i, sigma_i^2)
    # Score is the negative of the sample (higher = better for minimization)
    
    # Ensure sigma is positive
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Draw a single sample from the predictive posterior for each candidate
    # Use numpy's random number generator for efficiency
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Thompson sampling score: negative of the sample
    # We want to maximize this, so lower predicted values get higher scores
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
