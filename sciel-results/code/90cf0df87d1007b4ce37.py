"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()

# RNG for reproducible posterior sampling within a run
_rng = np.random.default_rng(42)


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Gaussian Process Lower Confidence Bound (GP-LCB).
    For minimization, we select the point with the lowest LCB value, which is
    equivalent to maximizing the score: score = -(mu - beta * sigma) = -mu + beta * sigma.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Check deadline: if we're running out of time, return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: exploit (lowest mu) with tiny exploration
        scores = -mu + 0.1 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Exploration parameter beta
    # A fixed beta is often robust for small search spaces.
    # We can also use a time-varying beta, but fixed is a good structural baseline.
    beta = 1.5
    
    # Score for maximization: -mu + beta * sigma
    # Higher score means more likely to be the minimum (exploit low mu) or high uncertainty (explore).
    scores = -mu + beta * sigma

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
