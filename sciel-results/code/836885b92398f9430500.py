"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: sample from the posterior GP at the candidate points.
    # We want to minimize f, so we look for the lowest sampled value.
    # To make "higher score = better", we return the negative of the sampled value.
    # Adding a tiny bit of noise to sigma prevents division by zero or issues if sigma is 0.
    sigma_safe = np.maximum(sigma, 1e-9)
    samples = mu + sigma_safe * np.random.randn(len(mu))
    return -samples
# EVOLVE-BLOCK-END
