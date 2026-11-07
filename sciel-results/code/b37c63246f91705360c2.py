# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization.
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # This directly maximizes the expected gain over the current best,
    # balancing exploration (high sigma) and exploitation (low mu).
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against degenerate / negative std (sigma is a standard deviation).
    s = np.clip(sigma, 1e-9, None)

    z = (f_best - mu) / s
    z = np.clip(z, -38.0, 38.0)  # keep CDF/PDF finite and well-behaved

    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)

    # Ensure non-negative scores (EI can be slightly negative due to numerical issues)
    ei = np.maximum(ei, 0.0)

    return ei
# EVOLVE-BLOCK-END
