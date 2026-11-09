"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling for minimization
    # Draw a single scalar sample from the standard normal distribution
    z = np.random.normal(0, 1)
    
    # The posterior predictive sample at each point is mu + sigma * z
    # Since we want to minimize the function, we want the point with the lowest sample
    # The acquisition score should be higher for points we prefer to evaluate
    # Therefore, we return the negative of the sample
    scores = -(mu + sigma * z)
    
    return scores
# EVOLVE-BLOCK-END
