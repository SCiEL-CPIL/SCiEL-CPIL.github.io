"""Bayesian optimization acquisition for Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Posterior-sampling-based acquisition function.

    Draws 50 latent function samples from the GP posterior and returns the
    maximum improvement over f_best across those samples, i.e. a Monte-Carlo
    estimate of Expected Improvement.
    """
    n = mu.shape[0]
    n_samples = 50

    # Ensure sigma is strictly positive to avoid division by zero.
    sigma_safe = np.maximum(sigma, 1e-12)

    # z-scores for the deterministic EI baseline
    z = (f_best - mu) / sigma_safe

    # Base EI (deterministic)
    ei_det = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Posterior sampling: for each sample, draw from N(mu, sigma)
    # We approximate the max over samples of improvement = max(0, f_best - sample)
    # Instead of drawing full vectors (which would be n_samples * n memory),
    # we draw one scalar per candidate across samples using a pre-generated
    # matrix of shape (n_samples, n).

    # To keep memory and time bounded, use a fixed number of samples.
    rng = np.random.RandomState(42)  # deterministic for reproducibility
    # Draw n_samples standard normal matrices of shape (n,)
    # Equivalent to: samples = mu[None, :] + sigma_safe[None, :] * Z
    # where Z has shape (n_samples, n)

    # For speed, draw Z in chunks if n is large, but n=2000 is fine.
    Z = rng.randn(n_samples, n)
    samples = mu[None, :] + sigma_safe[None, :] * Z  # (n_samples, n)

    # Improvement for each sample at each candidate: max(0, f_best - sample)
    improvement = np.maximum(0.0, f_best - samples)  # (n_samples, n)

    # MC estimate of EI: mean over samples
    scores = np.mean(improvement, axis=0)  # (n,)

    # Add a small exploration bonus: use the standard deviation of improvements
    # across samples as a proxy for uncertainty in the gain estimate.
    # This encourages exploration where the MC estimate is noisy.
    bonus = 0.1 * np.std(improvement, axis=0)

    scores = scores + bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
