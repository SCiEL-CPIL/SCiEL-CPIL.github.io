"""Acquisition function for Bayesian optimization of Michalewicz 2D using Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_TIMEOUT = 325.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of optimizing a deterministic acquisition function like EI or LCB,
    we sample a single function from the Gaussian Process posterior and select
    the point with the lowest sampled value.
    
    We return the negative of the sampled values so that the evaluator, which
    picks the candidate with the highest score, effectively picks the candidate
    with the lowest predicted function value.
    """
    # Deadline guard
    if time.time() - _T0 > _TIMEOUT:
        # Fallback: Pure Exploitation (pick lowest mu)
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a sample from the posterior distribution at the candidate points
    # The posterior is Gaussian with mean mu and std sigma
    noise = np.random.randn(len(mu))
    sampled_values = mu + sigma_safe * noise
    
    # We want to minimize the function, so the "best" sample is the lowest value.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled values.
    scores = -sampled_values
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
