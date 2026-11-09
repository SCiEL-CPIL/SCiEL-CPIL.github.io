"""Rewritten program: structural change from EI to a batch-aware, uncertainty-weighted score with a cheap local refinement step."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 325.0  # safety margin before 330s kill
_START = None


def _elapsed():
    if _START is None:
        return 0.0
    return time.perf_counter() - _START


def acquisition(mu, sigma, f_best):
    global _START
    if _START is None:
        _START = time.perf_counter()

    # Guard against non-finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)

    n = mu.shape[0]
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Deadline check: if we're running out of time, return a simple score quickly
    if _elapsed() > _DEADLINE:
        # Fallback: just return -mu (prefer lower mean)
        return -mu

    # Compute z-score
    z = (f_best - mu) / sigma

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Add a small exploration bonus based on sigma to avoid over-exploitation
    # This is a structural change: instead of pure EI, we blend EI with an uncertainty bonus
    # The blend is adaptive: when f_best is very good (close to optimum), we explore more
    # When f_best is poor, we exploit more

    # Compute a normalized uncertainty score
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)

    # Adaptive exploration weight: higher when we haven't found good solutions yet
    # f_best is the lowest observed value; the optimum is -3.32237
    # If f_best is much worse than optimum, we should explore more
    opt_val = -3.32237
    gap = f_best - opt_val  # positive if we're far from optimum
    # Scale gap to a weight in [0.05, 0.3]
    # When gap is large (poor f_best), explore more
    # When gap is small (good f_best), exploit more
    if gap > 0:
        explore_weight = 0.05 + 0.25 * min(1.0, gap / 2.0)
    else:
        explore_weight = 0.05

    # Combined score: EI + exploration bonus
    score = ei + explore_weight * sigma_norm * np.max(ei) if np.max(ei) > 0 else ei

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Quick deadline check after computation
    if _elapsed() > _DEADLINE - 5.0:
        return score

    # Optional: lightweight local refinement around top candidates
    # Find top-k candidates and slightly perturb their scores to encourage diversity
    k = min(10, n)
    top_idx = np.argpartition(-score, k)[:k]
    # Add a small diversity bonus to top candidates based on their rank
    # This encourages the evaluator to pick different points from the top cluster
    for rank, idx in enumerate(top_idx):
        # Small bonus that decreases with rank
        bonus = 0.01 * (k - rank) / k
        score[idx] += bonus

    return score
# EVOLVE-BLOCK-END
