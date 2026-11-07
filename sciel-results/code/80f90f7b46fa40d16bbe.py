"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling based acquisition for minimization.
    We sample from the posterior N(mu, sigma) and select the candidate
    with the lowest sampled value. To make this into a score where higher
    is better, we return -sampled_value.
    """
    # Ensure sigma is non-negative and handle numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior
    # We use a fixed random state? No, Thompson sampling should be stochastic.
    # However, the function is called in a loop. The randomness here introduces variance.
    # Standard Thompson Sampling: sample f(x) ~ N(mu, sigma^2) and pick argmin f(x).
    # Here we return scores. Higher score = evaluate next.
    # So score = -sampled_value.
    
    # To ensure we don't have infinite loops or issues, we just return the negative sample.
    # One draw per candidate.
    samples = mu + sigma * np.random.standard_normal(len(mu))
    
    # Return negative samples so that lower function values yield higher scores
    return -samples
# EVOLVE-BLOCK-END
