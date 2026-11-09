"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Sample from the posterior distribution
    # mu and sigma are arrays of shape (2000,)
    
    # Generate standard normal random numbers
    epsilon = np.random.normal(0.0, 1.0, size=mu.shape)
    
    # Calculate sampled function values
    # f_s(x) ~ N(mu(x), sigma(x)^2)
    sampled_values = mu + sigma * epsilon
    
    # We want to minimize the objective function.
    # The evaluator selects the candidate with the highest acquisition score.
    # Therefore, we return the negative of the sampled values so that
    # the minimum sampled value corresponds to the maximum score.
    scores = -sampled_values
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return scores
# EVOLVE-BLOCK-END
