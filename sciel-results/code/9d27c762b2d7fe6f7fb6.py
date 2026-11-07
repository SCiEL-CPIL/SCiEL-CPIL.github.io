"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a single sample from the posterior at each candidate
    # and return the negative sampled value (lower sampled value = higher score).
    # This stochastic approach balances exploration and exploitation differently
    # from deterministic EI/LCB by injecting randomness that can escape local optima.
    rng = np.random.default_rng()
    xi = rng.standard_normal(size=mu.shape)
    f_sample = mu + sigma * xi
    return -f_sample
# EVOLVE-BLOCK-END
