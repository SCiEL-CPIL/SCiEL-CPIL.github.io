"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Thompson Sampling: draw a single sample from the predictive posterior
    # For each candidate i, sample f_i ~ N(mu_i, sigma_i^2)
    # Score = -f_i (higher score means we expect lower objective value)
    # This naturally balances exploration (high sigma regions have more variance)
    # and exploitation (low mu regions tend to have lower samples)
    
    # Use a fixed random seed per call? No, we want different samples each step.
    # But we need reproducibility within a single acquisition call.
    # The evaluator calls acquisition once per step, so we just draw fresh samples.
    
    # Draw samples from N(mu, sigma^2)
    samples = mu + sigma * np.random.standard_normal(len(mu))
    
    # Score is negative of the sample (we want to minimize, so higher score = lower expected value)
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
