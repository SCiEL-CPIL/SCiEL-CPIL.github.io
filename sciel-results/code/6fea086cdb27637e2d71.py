"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE_SECONDS = 325.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) with posterior means in objective units
    - sigma: array of shape (2000,) with posterior standard deviations in objective units
    - f_best: lowest observed value so far
    
    Returns:
    - array of shape (2000,) with scores; higher means more promising
    """
    # Check deadline - if we're close to timeout, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Fallback: return negative mu (pure exploitation) to finish quickly
        return -mu

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)

    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe

    # Use stable computation to avoid overflow/underflow
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)

    ei = improvement * cdf + sigma_safe * pdf

    # UCB term for exploration
    # Adaptive beta: start with more exploration, decrease over time
    # Beale min is 0, max is ~22516 on [-4.5,4.5]^2
    # Use a simple heuristic: if f_best is still large, explore more
    if f_best > 500:
        beta = 3.0
    elif f_best > 50:
        beta = 2.5
    elif f_best > 10:
        beta = 2.0
    elif f_best > 1:
        beta = 1.5
    else:
        beta = 1.0

    ucb = -mu + beta * sigma_safe

    # Combine EI and UCB using a weighted geometric mean approach
    # This tends to find points that are good on both criteria
    # First, shift and scale to ensure positive values for geometric mean
    ei_min = np.min(ei)
    ucb_min = np.min(ucb)

    # Shift to make non-negative with small offset
    ei_shifted = ei - ei_min + 1e-6
    ucb_shifted = ucb - ucb_min + 1e-6

    # Normalize by max to get [0, 1] range
    ei_max = np.max(ei_shifted)
    ucb_max = np.max(ucb_shifted)
    if ei_max < 1e-9:
        ei_max = 1.0
    if ucb_max < 1e-9:
        ucb_max = 1.0

    ei_norm = ei_shifted / ei_max
    ucb_norm = ucb_shifted / ucb_max

    # Weighted geometric mean: balances exploitation and exploration
    # Use 0.5/0.5 for balanced exploration-exploitation
    w = 0.5
    score = ei_norm ** w * ucb_norm ** (1 - w)

    # Alternative: use a sum with adaptive weighting
    # More exploration early, more exploitation later
    if f_best > 50:
        alpha = 0.4  # More UCB weight
    elif f_best > 5:
        alpha = 0.5
    else:
        alpha = 0.6  # More EI weight

    score_alt = alpha * ei_norm + (1 - alpha) * ucb_norm

    # Use the maximum of both approaches
    score = np.maximum(score, score_alt)

    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
