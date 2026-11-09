"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a combination of Expected Improvement (EI) and an exploration bonus
    to balance exploitation and exploration.
    """
    t_start = time.time()
    deadline = 300.0  # safety margin: 300s total budget, return before 330s kill

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # Ensure EI is non-negative and finite
    ei = np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)

    # Check deadline
    if time.time() - t_start > deadline:
        return ei

    # Compute an exploration bonus based on uncertainty (sigma)
    # Scale sigma to [0, 1] range for the bonus
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    if sigma_max > sigma_min:
        sigma_scaled = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_scaled = np.zeros_like(sigma_safe)

    # Weight the exploration bonus
    # The bonus should be significant when uncertainty is high but not dominate
    # over the expected improvement for obvious improvements
    exploration_weight = 0.1  # small weight for exploration
    bonus = exploration_weight * sigma_scaled

    # Combine EI and bonus
    scores = ei + bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline again
    if time.time() - t_start > deadline:
        return ei

    return scores
# EVOLVE-BLOCK-END
