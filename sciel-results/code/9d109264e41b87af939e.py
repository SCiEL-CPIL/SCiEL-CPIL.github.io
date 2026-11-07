"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: draw a single posterior sample and minimize
    # normalized regret (theta - f_best) / sqrt(cost).
    # Use a fixed-seed RNG for determinism within the acquisition call.
    rng = np.random.RandomState(42)
    z = rng.standard_normal(mu.shape)
    theta = mu + sigma * z
    # Normalized regret: lower is better, so we negate for higher-is-better scoring
    # and divide by sqrt(cost) to account for evaluation cost.
    cost_safe = np.maximum(cost, 1e-9)
    score = -(theta - f_best) / np.sqrt(cost_safe)
    return score
# EVOLVE-BLOCK-END
