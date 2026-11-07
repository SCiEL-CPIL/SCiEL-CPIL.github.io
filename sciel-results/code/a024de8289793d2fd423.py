"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviations (2000,).
        f_best: Best observed value so far.
        
    Returns:
        Array of scores (2000,), where higher score means better candidate.
    """
    # Generate standard normal random variables
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior distribution
    # y_i ~ N(mu_i, sigma_i^2)
    samples = mu + sigma * z
    
    # We want to minimize the function value, so we maximize the negative sample
    # The candidate with the lowest sampled value will have the highest score
    scores = -samples
    
    return scores
# EVOLVE-BLOCK-END
