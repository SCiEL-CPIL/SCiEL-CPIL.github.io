"""Rewritten cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Composite cost-aware acquisition: Blends normalized EI and PI.
    
    Strategy:
    1. Compute Expected Improvement (EI) and Probability of Improvement (PI).
    2. Normalize both to [0, 1] range based on current batch statistics.
    3. Blend them (50/50) to balance exploitation (EI) and exploration (PI).
    4. Scale by 1/cost to account for evaluation cost.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # safety margin before 300s limit

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # z-score for improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)

    # Normalize EI and PI to [0, 1] for blending
    # Use max-min normalization with a small epsilon to avoid division by zero
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range

    pi_min = np.min(pi)
    pi_max = np.max(pi)
    pi_range = pi_max - pi_min
    if pi_range < 1e-12:
        pi_norm = np.zeros_like(pi)
    else:
        pi_norm = (pi - pi_min) / pi_range

    # Blend EI and PI
    # 50% weight to each. PI helps explore uncertain regions where EI might be low
    # due to high sigma but high potential.
    blended = 0.5 * ei_norm + 0.5 * pi_norm

    # Cost-aware score: Blended score per unit cost
    score = blended / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard: if we're close to deadline, return immediately
    if time.time() > deadline - 5.0:
        return score

    return score
# EVOLVE-BLOCK-END
