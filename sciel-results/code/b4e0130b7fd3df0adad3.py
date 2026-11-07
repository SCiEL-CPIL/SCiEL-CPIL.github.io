"""Acquisition for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 295.0  # safety margin before the 330s kill


def acquisition(mu, sigma, f_best):
    """Hybrid PI + conditional exploration bonus.
    
    Uses Probability of Improvement as the primary signal, which is well-suited
    for Branin 2D's multimodal landscape. Adds a conditional exploration bonus
    that scales with both uncertainty and the gap between predicted value and
    f_best, creating a smoother exploration-exploitation tradeoff than a flat
    sigma bonus.
    
    The conditional bonus is structured as: bonus = w * sigma * exp(-alpha * max(0, mu - f_best))
    This rewards uncertain regions that are NOT already clearly worse than f_best,
    while dampening exploration in regions where the mean is already far above f_best.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return -mu

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Ensure sigma is positive and finite
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # Primary signal: Probability of Improvement
    z = (f_best - mu) / sigma_safe
    pi_scores = norm.cdf(z)

    # Conditional exploration bonus:
    # Reward uncertainty in regions where the prediction is not clearly worse than f_best.
    # The exponential decay dampens the bonus when mu is far above f_best.
    gap = np.maximum(0.0, mu - f_best)
    alpha = 0.5  # decay rate for the gap
    exploration_weight = 0.08
    bonus = exploration_weight * sigma_safe * np.exp(-alpha * gap)

    # Normalize bonus to be comparable to PI scale
    bonus_max = np.max(bonus)
    if bonus_max > 1e-12:
        bonus = bonus / bonus_max * 0.1  # cap the bonus contribution at 0.1
    else:
        bonus = np.zeros_like(bonus)

    # Combine
    scores = pi_scores + bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
