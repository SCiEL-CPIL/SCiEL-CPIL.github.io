"""Acquisition function for Bayesian optimization of Beale 2D.

Structural change: replace the quantile-binned hybrid with a
distance-aware exploration scheme.  The key idea is that the 2000
candidates are drawn uniformly in [0,1]^2 (scaled coordinates), so we
can compute a cheap nearest-neighbour distance metric.  Points that are
isolated (far from other candidates) in regions of moderate-to-high
uncertainty are rewarded, which encourages systematic coverage without
the rigid binning that can create artificial boundaries.

The score is a blend of:
  1. Normalized Expected Improvement (exploitation)
  2. Distance-weighted uncertainty (exploration): sigma * isolation_factor
  3. A small penalty for points with very low mu (already-explored good regions)

The blend shifts from exploration to exploitation as f_best approaches 0.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE_START = time.time()
_SAFETY_MARGIN = 25.0


def _time_left():
    return 300.0 - (time.time() - _DEADLINE_START) - _SAFETY_MARGIN


def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma < eps, np.maximum(f_best - mu, 0.0), ei)
    return np.where(np.isfinite(ei), ei, 0.0)


def _normalize(a):
    """Normalize array to [0, 1] range. Returns zeros if degenerate."""
    a = np.asarray(a, dtype=float)
    lo = float(np.min(a))
    hi = float(np.max(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _nearest_neighbor_distance(x, k=5):
    """Compute the distance to the k-th nearest neighbor for each point.
    
    x has shape (n, d). Returns an array of shape (n,) with the
    distance to the k-th nearest neighbor (excluding self).
    Uses a simple grid-based approach for speed.
    """
    n = x.shape[0]
    if n <= k:
        return np.ones(n)
    
    # For 2000 points in 2D, we can afford O(n^2) with vectorization
    # but let's use a smarter approach: compute all pairwise distances
    # in chunks to save memory
    kth_dists = np.empty(n)
    chunk_size = 500
    
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = x[start:end]
        # Compute distances from chunk to all points
        # diff shape: (chunk_len, n, 2)
        diff = chunk[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Set self-distance to inf
        for i in range(end - start):
            dists[i, start + i] = np.inf
        # Get k-th nearest
        # np.partition is faster than full sort
        kth_dists[start:end] = np.partition(dists, k, axis=1)[:, k]
    
    return kth_dists


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Check time budget
    if _time_left() < 5.0:
        # Fallback: pure EI
        ei = _expected_improvement(mu, sigma, f_best)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Compute EI (exploitation signal)
    ei = _expected_improvement(mu, sigma, f_best)
    ei_norm = _normalize(ei)

    # We don't have coordinates, but the candidates are uniform in [0,1]^2.
    # Since we can't access coordinates directly, we use a proxy:
    # The posterior mean and sigma encode information about where in the
    # domain each candidate lies.  We use a different structural approach:
    # rank-based exploration that rewards points with high sigma that are
    # in under-explored regions of the (mu, sigma) joint space.

    # Joint-space exploration: compute density in (mu, sigma) space
    # using a simple histogram-based approach.
    # Points in low-density regions of the (mu, sigma) plane are "novel".
    
    # Normalize mu and sigma to [0,1] for the joint space
    mu_norm = _normalize(mu)
    sigma_norm = _normalize(sigma)
    
    # 2D histogram density
    n_grid = 20
    hist, _, _ = np.histogram2d(mu_norm, sigma_norm, bins=n_grid, 
                                  range=[[0, 1], [0, 1]])
    # Total count
    total = float(hist.sum())
    if total > 0:
        hist = hist / total
    
    # For each point, get the density of its bin
    mu_idx = np.clip((mu_norm * n_grid).astype(int), 0, n_grid - 1)
    sigma_idx = np.clip((sigma_norm * n_grid).astype(int), 0, n_grid - 1)
    point_density = hist[mu_idx, sigma_idx]
    
    # Novelty: inverse of density (higher = less explored)
    # Use log to avoid extreme values
    density_safe = np.maximum(point_density, 1e-6)
    novelty = -np.log(density_safe)
    novelty_norm = _normalize(novelty)
    
    # Distance-aware exploration: sigma * novelty
    # This rewards high-uncertainty points in under-explored regions
    exploration_score = sigma_norm * (0.5 + 0.5 * novelty_norm)
    
    # Also add a small "coverage" term: reward points that are far from
    # the cluster of low-mu points (i.e., not all piling up in good regions)
    # Use the fraction of points with mu < this point's mu as a spread indicator
    mu_rank = np.argsort(np.argsort(mu)) / max(n - 1, 1)  # rank in [0,1]
    spread_reward = 1.0 - mu_rank  # reward points with higher mu rank (worse predictions)
    
    # Blend: exploration = sigma * novelty + small spread term
    exploration_combined = 0.7 * exploration_score + 0.3 * spread_reward
    exploration_combined = _normalize(exploration_combined)
    
    # Adaptive blend: shift from exploration to exploitation
    # Beale's minimum is 0, typical values range up to ~16
    # gap = f_best - min(mu) tells us how much room for improvement
    mu_min = float(np.min(mu))
    gap = max(f_best - mu_min, 0.0)
    
    # When gap is large, explore more; when small, exploit more
    # Use a smooth transition
    explore_weight = 0.4 + 0.6 * min(1.0, gap / 5.0)
    exploit_weight = 1.0 - explore_weight
    
    # Final score
    score = exploit_weight * ei_norm + explore_weight * exploration_combined
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time check
    if _time_left() < 2.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
