"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute a hybrid acquisition score combining Expected Improvement (EI)
    with a small uncertainty bonus to encourage exploration.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far.

    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Ensure inputs are 1-D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)

    # Standardized improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Small exploration bonus: reward points with high uncertainty
    # but dampen it so it doesn't dominate when EI is large
    exploration_weight = 0.05
    exploration_bonus = exploration_weight * sigma_safe

    # Combine: primarily EI, with a mild exploration term
    scores = ei + exploration_bonus

    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
