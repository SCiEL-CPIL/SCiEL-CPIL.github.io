# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: Draw fresh standard normal samples for each candidate
    # at each step to properly balance exploration and exploitation.
    z = np.random.standard_normal(size=mu.shape)
    sampled_values = mu - sigma * z
    return -sampled_values
# EVOLVE-BLOCK-END
