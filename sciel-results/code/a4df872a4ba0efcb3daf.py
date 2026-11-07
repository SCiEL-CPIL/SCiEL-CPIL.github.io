"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Quantile Acquisition Function (QAF):
    # Maximizes the probability that f(x) < f_best, i.e., P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    # Since we want higher scores to mean "evaluate next", we return this probability directly.
    # We add a small epsilon to sigma to avoid division by zero.
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    scores = norm.cdf(z)
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
