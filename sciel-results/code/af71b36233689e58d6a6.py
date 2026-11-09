"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
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
    Compute acquisition scores for 2000 candidate points.
    Uses posterior sampling to approximate Expected Improvement for minimization.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Check deadline: if we're running out of time, just return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: exploit (lowest mu) with tiny exploration
        scores = -mu + 0.1 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Posterior sampling-based acquisition
    # Draw 50 samples from the GP posterior at the candidate points
    n_samples = 50
    n_candidates = mu.shape[0]
    
    # Generate random samples: y_sample = mu + sigma * z where z ~ N(0,1)
    # Use vectorized operations for speed
    rng = np.random.default_rng()
    # Shape: (n_samples, n_candidates)
    z_samples = rng.standard_normal((n_samples, n_candidates))
    y_samples = mu[None, :] + sigma[None, :] * z_samples  # (n_samples, n_candidates)
    
    # For each candidate, compute the maximum improvement over f_best across all samples
    # improvement = max(f_best - y_sample, 0) for each sample, then take mean or max
    # We want to estimate E[max(f_best - Y, 0)] where Y ~ N(mu, sigma^2)
    # Approximate by: for each candidate, compute mean of max(f_best - y_i, 0) over samples
    improvements = np.maximum(f_best - y_samples, 0.0)  # (n_samples, n_candidates)
    ei_approx = improvements.mean(axis=0)  # (n_candidates,)
    
    # Ensure all scores are finite
    scores = ei_approx
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
