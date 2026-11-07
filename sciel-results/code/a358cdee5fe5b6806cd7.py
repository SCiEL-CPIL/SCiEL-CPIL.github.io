"""Acquisition function for Bayesian optimization of Beale 2D.

Structural change: Posterior Mean Ranking.
Scores candidates by their rank in ascending order of posterior mean (mu),
treating the 2000 candidates as a discrete distribution. This forces
aggressive exploitation of the current best predicted region while using
a small, adaptive uncertainty floor to break ties among near-identical
mu values. The key insight: for Beale's narrow valley, pure exploitation
of the lowest predicted mean converges faster than uncertainty-weighted
schemes, and the rank-based formulation is scale-invariant and robust to
posterior calibration issues.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_START = time.time()
_SAFETY_MARGIN = 25.0


def _time_left():
    return 300.0 - (time.time() - _DEADLINE_START) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety: ensure finite values.
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Posterior Mean Ranking:
    # 1. Compute rank of each point in ascending order of mu.
    #    Lower mu (better predicted value) gets higher rank score.
    #    Using argsort of argsort gives 0-based rank where 0 = lowest mu.
    # 2. Score = (n - rank) / n, so best mu gets score ~1.0, worst gets ~0.
    # 3. Add a small adaptive uncertainty bonus to break ties among
    #    points with nearly identical mu, preventing degenerate selection
    #    while keeping exploitation dominant.
    
    # Compute ranks: rank[i] = number of points with mu < mu[i]
    # Using the "order of order" trick for O(n log n) ranking.
    order = np.argsort(mu, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    
    # Rank score: higher = better (lower mu).
    # Normalize to [0, 1]: best mu (rank 0) -> 1.0, worst mu (rank n-1) -> 0.0
    rank_score = (n - 1 - ranks) / max(n - 1, 1)
    
    # Adaptive uncertainty floor:
    # The bonus is scaled by the relative spread of mu values.
    # When mu values are tightly clustered (late optimization), the bonus
    # is small, maintaining aggressive exploitation.
    # When mu values are spread out (early optimization), the bonus is
    # larger, allowing some exploration among uncertain regions.
    mu_range = float(np.max(mu) - np.min(mu))
    if mu_range < 1e-12:
        # All mu values are essentially identical; use sigma to break ties.
        sigma_norm = sigma / max(np.max(sigma), 1e-12)
        score = 0.5 + 0.5 * sigma_norm
    else:
        # Scale the uncertainty bonus by how "tight" the mu distribution is.
        # tightness = 1 / (1 + mu_range / typical_scale)
        # typical_scale for Beale is ~1.0 (range ~16, but local features ~1)
        typical_scale = 1.0
        tightness = 1.0 / (1.0 + mu_range / typical_scale)
        
        # Uncertainty bonus: normalized sigma, scaled by tightness.
        # Max bonus is 0.1 * tightness, so at most 10% of the score range.
        sigma_max = float(np.max(sigma))
        if sigma_max > 1e-12:
            sigma_norm = sigma / sigma_max
        else:
            sigma_norm = np.zeros(n)
        
        uncertainty_bonus = 0.1 * tightness * sigma_norm
        
        # Final score: dominant rank component + small uncertainty bonus.
        score = rank_score + uncertainty_bonus
    
    # Final numerical safety: finite, higher = evaluate next.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
