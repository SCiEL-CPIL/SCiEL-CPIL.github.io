"""Cost-aware acquisition for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE_MARGIN_S = 5.0  # safety margin before the 330s kill
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _deadline_reached():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN_S)


def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are close to the wall-clock limit, return a safe finite score
    # that picks the lowest-cost candidate (cheapest to evaluate) to avoid wasting time.
    if _deadline_reached():
        # Prefer lowest cost; break ties by lower mu (optimistic exploitation)
        idx = np.argmin(cost)
        scores = np.full_like(mu, -1e30, dtype=np.float64)
        scores[idx] = 0.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)

    # UCB-based acquisition: balances exploration and exploitation
    # beta controls exploration; we use a moderate value
    beta = 2.0

    # Compute UCB score
    ucb = mu + beta * sigma

    # Cost-aware weighting: penalize high cost
    # Use cost^1.5 to bias toward low-cost regions (measured improvement)
    cost_penalty = np.power(cost, 1.5)

    # Normalize: we want higher scores to be better
    # Since we're minimizing, lower mu is better; UCB with +beta*sigma rewards uncertainty
    # But we want to minimize, so we should use -mu or (f_best - mu) style
    # Actually, for minimization, a good acquisition is: (f_best - mu) + beta * sigma
    # This rewards points that are likely below f_best and have high uncertainty

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB for minimization:
    # UCB = (f_best - mu) + beta * sigma  (optimistic: assumes we can get lower by beta*sigma)
    ucb_min = (f_best - mu) + beta * sigma

    # Combine: use a blend or just UCB which avoids EI saturation in low-noise regimes
    # The knowledge suggests UCB with beta=2 works well
    score = ucb_min / cost_penalty

    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e30)

    # Clip to avoid extreme values causing issues
    score = np.clip(score, -1e10, 1e10)

    return score
# EVOLVE-BLOCK-END
