"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard variables
# The constraint is 330s wall-clock for the whole program.
# The acquisition function is called 25 times (30 evals - 5 initial).
# Each call must be fast. Thompson Sampling involves drawing 2000 samples.
# This is O(N) and very fast in NumPy.

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling Acquisition Function.
    
    Instead of computing a deterministic score like EI or UCB, we sample 
    a function value from the posterior distribution at each candidate point 
    and select the point with the lowest sampled value.
    
    Since the task requires returning scores where higher is better, 
    we return the negative of the sampled value.
    
    Posterior: y ~ N(mu, sigma^2)
    We want to minimize y.
    So we sample y_hat ~ N(mu, sigma) and return -y_hat.
    The evaluator picks argmax(-y_hat) which is argmin(y_hat).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    
    # Draw a single sample from the posterior for all 2000 candidates
    # np.random.normal(loc=mu, scale=sigma)
    # Using standard normal and scaling is often slightly faster or equivalent
    z = np.random.standard_normal(mu.shape)
    samples = mu + sigma * z
    
    # We want to minimize the objective.
    # The evaluator picks the candidate with the highest score.
    # So we return negative samples.
    scores = -samples
    
    # Safety: ensure finite
    scores[~np.isfinite(scores)] = -np.inf
    
    return scores

# EVOLVE-BLOCK-END
