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
    Compute acquisition scores for 2000 candidate points using posterior sampling.
    Draws multiple samples from the GP posterior and averages the improvement
    over f_best across samples to approximate Expected Improvement.
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

    # Number of posterior samples
    n_samples = 50

    # Draw posterior samples: Y_s ~ N(mu, sigma^2) for s in 1..n_samples
    # Shape: (n_samples, n_candidates)
    samples = _rng.normal(
        loc=mu[None, :],
        scale=sigma[None, :],
        size=(n_samples, mu.size)
    )

    # For each sample, compute improvement over f_best
    # improvement_s = max(f_best - Y_s, 0)
    improvements = np.maximum(f_best - samples, 0.0)

    # Average improvement across samples = estimate of EI
    scores = np.mean(improvements, axis=0)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
