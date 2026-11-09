"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # safety margin before 330s kill
_EPS = 1e-9


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores; higher => evaluate next.

    Uses Expected Improvement with a diversity-preserving repulsive
    penalty in the (mu, sigma) feature space to avoid clustering
    around a single local minimum in the multimodal Hartmann 3D landscape.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, _EPS)

    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Diversity penalty: repulsive force in (mu, sigma) space
    # Scale features to unit variance for stable kernel
    mu_scale = np.std(mu) + _EPS
    sigma_scale = np.std(sigma) + _EPS
    mu_scaled = mu / mu_scale
    sigma_scaled = sigma / sigma_scale

    # Compute pairwise squared distances in 2D (mu_scaled, sigma_scaled)
    # Use chunked computation to manage memory
    n = len(mu)
    diversity_penalty = np.zeros(n, dtype=np.float64)
    
    # Gaussian kernel bandwidth
    bandwidth = 0.3
    
    # Compute in chunks to avoid O(n^2) memory
    chunk_size = 500
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # Distance from candidates [i_start:i_end] to all candidates
        diff_mu = mu_scaled[i_start:i_end, None] - mu_scaled[None, :]
        diff_sigma = sigma_scaled[i_start:i_end, None] - sigma_scaled[None, :]
        sq_dist = diff_mu ** 2 + diff_sigma ** 2
        # Gaussian kernel
        kernel = np.exp(-sq_dist / (2.0 * bandwidth ** 2))
        # Sum over all j (including self, but self-kernel is 1)
        diversity_penalty[i_start:i_end] = kernel.sum(axis=1)

    # Normalize penalty: higher density => larger penalty
    # Scale so that the penalty is meaningful relative to EI
    ei_range = ei.max() - ei.min() + _EPS
    penalty_weight = 0.15 * ei_range
    scores = ei - penalty_weight * diversity_penalty

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        z2 = (f_best - mu) / sigma
        z2 = np.clip(z2, -10.0, 10.0)
        scores = np.maximum((f_best - mu) * norm.cdf(z2) + sigma * norm.pdf(z2), 0.0)
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
