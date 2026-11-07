"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile Acquisition Function (QAF) with a small exploration bonus.
    Maximizes the probability of finding a value below f_best,
    P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)

    # Compute the probability of improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    prob_improvement = norm.cdf(z)

    # To avoid purely exploitative behavior early on, add a small
    # uncertainty-based exploration term. This helps escape local optima
    # in the surrogate. We weight it lightly so the main driver is
    # still the probability of a new minimum.
    #
    # Use a small multiple of sigma to encourage exploration in
    # high-uncertainty regions, but scaled down so it doesn't dominate.
    exploration_weight = 0.01
    score = prob_improvement + exploration_weight * (sigma_safe / (np.max(sigma_safe) + 1e-12))

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)

    return score
# EVOLVE-BLOCK-END
