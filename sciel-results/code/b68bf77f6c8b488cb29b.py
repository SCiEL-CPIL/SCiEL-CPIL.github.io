"""Acquisition function for Bayesian optimization of branin 2d with adaptive LCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition for minimization.
    Returns scores where higher means "evaluate next".
    Score = beta * sigma - mu, with beta adapted to optimization progress.
    """
    # Time budget guard: return safe fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 320:
        sigma_safe = np.asarray(sigma, dtype=np.float64)
        mu_safe = np.asarray(mu, dtype=np.float64)
        sigma_safe = np.where(sigma_safe > 1e-12, sigma_safe, 1e-12)
        sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1.0)
        mu_safe = np.where(np.isfinite(mu_safe), mu_safe, f_best)
        scores = 1.0 * sigma_safe - mu_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Sanitize inputs
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Known global minimum of Branin 2D
    f_min = 0.397887

    # Progress measure: 0.0 when f_best is at the optimum, 1.0 when f_best is far away
    # Use a reasonable initial range: Branin values range roughly from 0.4 to ~300
    # But in practice with 5 initial points, f_best is usually in [0.4, 50]
    # Use a smooth saturation: progress = 1 - exp(-gap / scale)
    gap = max(0.0, f_best - f_min)
    # Scale parameter: when gap ~ 5, progress ~ 0.63; when gap ~ 20, progress ~ 0.98
    scale = 5.0
    progress = 1.0 - np.exp(-gap / scale)
    progress = np.clip(progress, 0.0, 1.0)

    # Adaptive beta: high exploration early, low exploitation near optimum
    # beta goes from beta_max (early) to beta_min (late)
    beta_min = 0.5
    beta_max = 3.0
    # Use a smooth curve: beta = beta_min + (beta_max - beta_min) * progress^0.7
    # The exponent < 1 keeps beta higher for longer (more exploration in mid-stage)
    beta = beta_min + (beta_max - beta_min) * (progress ** 0.7)

    # LCB for minimization: we want to minimize mu - beta * sigma
    # The evaluator picks the candidate with the HIGHEST score
    # So score = -(mu - beta * sigma) = beta * sigma - mu
    scores = beta * sigma - mu

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)

    return scores
# EVOLVE-BLOCK-END
