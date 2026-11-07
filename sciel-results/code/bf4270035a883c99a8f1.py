"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI).
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Use a small epsilon to avoid division by zero
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    # Clip z to a reasonable range to avoid numerical issues
    z = np.clip(z, -10, 10)
    return norm.cdf(z)
# EVOLVE-BLOCK-END
