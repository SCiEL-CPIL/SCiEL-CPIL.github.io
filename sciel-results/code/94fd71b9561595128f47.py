"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a single sample from the posterior at each candidate point
    # and return the negative sample as the score (higher score = more likely to be evaluated).
    # Since we want to minimize, lower samples are better, so we negate them for scoring.
    
    # Ensure sigma is non-negative and handle any numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from N(mu, sigma) for each of the 2000 candidates
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Return negative samples so that lower objective values get higher scores
    return -samples
# EVOLVE-BLOCK-END
