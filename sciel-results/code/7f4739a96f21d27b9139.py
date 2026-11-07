"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 280.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)

    # z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)

    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)

    # Exploration Bonus:
    # Rastrigin is multimodal. Pure EI can be too greedy if the GP
    # overfits to local minima. Adding a moderate multiple of sigma
    # encourages visiting high-uncertainty areas.
    exploration_bonus = 0.5 * sigma

    # Total Score: Regularized EI divided by cost
    total_score = (ei + exploration_bonus) / np.maximum(cost, 1e-9)

    # Ensure finite output
    return np.nan_to_num(total_score, nan=0.0, posinf=0.0, neginf=0.0)
# EVOLVE-BLOCK-END
