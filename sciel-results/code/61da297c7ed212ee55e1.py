"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement plus a bounded smooth exploration bonus.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe

    # EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Bounded exploration term: scales with uncertainty but saturates so it
    # does not dominate exploitation.  The denominator (sigma + 0.1) keeps the
    # bonus O(1) and makes it negligible when sigma is small.
    explore = sigma_safe / (sigma_safe + 0.1)

    # Combine: primarily cost-aware EI, with a small smooth exploration push.
    score = (ei + 0.05 * explore) / np.maximum(cost, eps)

    # Ensure finite output as required.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
