import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization."""
    t_start = time.time()
    deadline = 330.0 - 20.0  # 20 second safety margin

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma and cost to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)

    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware scoring: EI / cost^0.5 with exploration bonus
    # Using sqrt(cost) instead of linear cost gives more exploration
    # in cheap regions while still preferring high improvement
    
    # Primary score: EI / sqrt(cost) - balances improvement vs cost
    score = ei / np.sqrt(cost_safe)

    # Exploration term: UCB-like with sigma / sqrt(cost)
    # This encourages visiting uncertain regions, weighted by cost
    ucb_weight = 0.15
    ucb_term = ucb_weight * sigma_safe / np.sqrt(cost_safe)
    
    # Additional diversity term: encourage visiting low-cost regions
    # when EI is small (converged state)
    ei_max = np.max(ei) if len(ei) > 0 else 0.0
    ei_mean = np.mean(ei) if len(ei) > 0 else 0.0
    
    if ei_max > 1e-10:
        convergence_ratio = ei_mean / (ei_max + 1e-12)
        # When converged, increase exploration weight
        exploration_boost = 0.1 + 0.2 * (1.0 - np.clip(convergence_ratio, 0, 1))
    else:
        exploration_boost = 0.3

    # Boost UCB weight when converged
    ucb_weight = 0.1 + exploration_boost
    ucb_term = ucb_weight * sigma_safe / np.sqrt(cost_safe)

    total_score = score + ucb_term

    # Ensure finite output
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)

    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > deadline:
        return np.zeros_like(total_score)

    return total_score
