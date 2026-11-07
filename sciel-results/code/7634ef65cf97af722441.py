"""Cost-aware acquisition function for Bayesian optimization."""
import numpy as np
from scipy.stats import norm
import time

# --- deadline guard ---
_START = time.time()
_LIMIT = 315.0  # 15 s margin before 330 s


def _elapsed():
    return time.time() - _START


def _time_out():
    return _elapsed() > _LIMIT


def acquisition(mu, sigma, f_best, cost):
    """
    Returns 2000 finite scores (higher = evaluate next).
    mu, sigma, f_best, cost have shape (2000,).
    """
    if _time_out():
        # Safety: return a simple, fast score to avoid timeout
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # --- Core acquisition: cost-aware Expected Improvement (EI) ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z),  z = (f_best - mu)/sigma
    # Cost-aware: divide by cost to reward cheap points.
    # Add a small UCB term for exploration in early iterations.

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe

    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB term: beta * sigma, where beta adapts to how far we've explored.
    # Use cost as a proxy: early on cost is low, so we can afford more exploration.
    # beta decays as total cost increases.
    mean_cost = np.mean(cost)
    # Normalize cost: in the loop, cost starts around 1-2 and goes up to ~30.
    # beta goes from ~2.0 early to ~0.5 late.
    beta = 2.0 / (1.0 + mean_cost / 5.0)

    ucb = beta * sigma

    # Combine: EI/cost + UCB/cost
    # Use a weighted blend: primarily EI/cost, plus UCB/cost for exploration.
    score_ei = ei / (cost + eps)
    score_ucb = ucb / (cost + eps)

    # Blend: 80% EI, 20% UCB
    score = 0.8 * score_ei + 0.2 * score_ucb

    # Ensure all finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
