"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Uses Expected Improvement per unit cost (BOB) with a UCB exploration blend,
    where the exploration weight decays as f_best approaches 0.
    """
    t0 = time.time()

    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: EI divided by cost (BOB - Bayesian Optimization under Budget)
    ei_cost = ei / cost_safe

    # UCB component for exploration
    # beta decays as f_best gets smaller (search matures)
    # f_best for griewank starts around 0.01-0.1 and goes toward 0
    progress = np.clip(1.0 - f_best, 0.0, 1.0)
    beta = 0.5 * (1.0 - progress) + 0.05
    ucb = mu + beta * sigma_safe
    # For minimization, lower UCB is better, so we use negative UCB as score
    ucb_score = -ucb / cost_safe

    # Blend: weight EI/cost more heavily, add some UCB for exploration
    # As f_best decreases, rely more on exploitation (EI/cost)
    ei_weight = 0.7 + 0.3 * progress
    ucb_weight = 1.0 - ei_weight

    # Normalize ucb_score to be comparable to ei_cost
    # ucb_score can be negative; shift so that the max is 0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > -1e-12 else 0.0
    if ucb_max < 0:
        ucb_shifted = ucb_score - ucb_max  # now <= 0, max is 0
    else:
        ucb_shifted = ucb_score

    # Scale ucb to be in similar range as ei_cost
    ei_max = np.max(ei_cost) if np.max(ei_cost) > 1e-12 else 1.0
    ucb_scaled = ucb_shifted / ei_max if ei_max > 1e-12 else ucb_shifted

    # Combined score
    score = ei_weight * ei_cost + ucb_weight * ucb_scaled

    # Clamp non-finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Ensure non-negative scores
    score_min = np.min(score)
    if score_min < 0:
        score = score - score_min

    # Deadline guard
    if time.time() - t0 > 280:
        return score

    return score
# EVOLVE-BLOCK-END
