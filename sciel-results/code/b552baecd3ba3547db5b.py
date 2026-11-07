"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 285.0

# Track call count for adaptive Thompson Sampling
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses Thompson Sampling: sample from the predictive posterior for each
    candidate and use the negative sample as the score. This naturally balances
    exploration and exploitation.
    """
    global _call_count
    _call_count += 1

    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Thompson Sampling: draw a single sample from each predictive posterior
    # s_i ~ N(mu_i, sigma_i^2), score = -s_i (higher is better)
    # Use a fixed random state per call to ensure reproducibility within a run
    rng = np.random.RandomState(_call_count)
    samples = mu + sigma * rng.standard_normal(n)
    
    # Score is the negative of the sample (we want to minimize)
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
