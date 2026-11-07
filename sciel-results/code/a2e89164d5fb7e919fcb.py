"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Entropy maximization proxy: log(sigma^2) = 2*log(sigma)
    # This directly targets regions of highest posterior variance,
    # forcing the sampler to map the complex multimodal landscape
    # of the Beale function more efficiently than mean-biased methods.
    # Add a small constant to ensure finite values even if sigma is 0.
    sigma_safe = np.maximum(sigma, 1e-10)
    scores = 2.0 * np.log(sigma_safe)
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
