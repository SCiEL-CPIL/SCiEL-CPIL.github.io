"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    stronger exploration term to handle the high dimensionality (6D)
    of the Hartmann function.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)

    # Standardized improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    # Using norm.cdf and norm.pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Exploration term: Increased weight to encourage exploration in 6D
    # The previous weight of 0.1 was too low, leading to potential local optima trapping.
    # A weight of 1.0 provides a stronger UCB-like behavior while retaining EI's exploitation.
    exploration_weight = 1.0
    scores = ei + exploration_weight * sigma_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
