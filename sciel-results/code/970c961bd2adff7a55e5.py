"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe fallback if we're getting close to the limit
    elapsed = time.time() - _start_time
    if elapsed > 280:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -30, 30)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Posterior lower bound (exploration)
    # For minimization, LCB = mu - beta * sigma
    # We want to maximize the score, so we use: -LCB = -mu + beta * sigma
    # But we also want to account for improvement over f_best

    # Cost-aware Expected Improvement
    score_ei = ei / cost_safe

    # Cost-aware LCB-based score
    # Higher score means more promising (lower predicted value, higher uncertainty)
    # LCB score: (beta * sigma - mu) / cost
    # This encourages exploring high-uncertainty areas with low predicted values
    beta = 2.0
    score_lcb = (beta * sigma_safe - mu) / cost_safe

    # Combine: weighted sum of EI/cost and LCB/cost
    # Use a moderate weight for exploration
    w_ei = 0.6
    w_lcb = 0.4

    score = w_ei * score_ei + w_lcb * score_lcb

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
