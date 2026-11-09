"""Rewritten program for cost-aware acquisition (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, fall back to simple greedy
    if time.time() - _START_TIME > _TIME_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_arr, 1e-6)
        return -mu_arr / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    cost = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)

    # f_best is the lowest observed value
    fb = float(f_best)

    # 1. Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (fb - mu) / sigma_safe
    # Clip z to avoid numerical issues in extreme tails
    z_clipped = np.clip(z, -12, 12)

    ei = (fb - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # EI should be non-negative; clip small negatives from numerical error
    ei = np.maximum(ei, 0.0)

    # 2. Compute UCB-style exploration score for minimization
    # We want to maximize the score.
    # Optimistic value: mu - beta * sigma
    # Gain relative to f_best: f_best - (mu - beta * sigma) = f_best - mu + beta * sigma

    # Adaptive beta:
    # When f_best is large (e.g., > 5), use higher beta for exploration.
    # When f_best is small (e.g., < 1), use lower beta for exploitation.
    # Rastrigin values: global min 0, local mins ~8-10.
    # Use a smooth transition based on f_best.

    # Normalize f_best for weight calculation
    # t=0 when fb <= 0, t=1 when fb >= 10
    t = np.clip(fb / 10.0, 0.0, 1.0)

    # Beta ranges from 1.0 (exploit) to 2.5 (explore)
    beta = 1.0 + 1.5 * t

    ucb_gain = fb - mu + beta * sigma_safe

    # 3. Blend EI and UCB
    # When fb is low (t=0), w_ucb = 0, w_ei = 1 -> Pure EI
    # When fb is high (t=1), w_ucb = 0.7, w_ei = 0.3 -> More exploratory
    w_ucb = 0.7 * t
    w_ei = 1.0 - w_ucb

    raw_score = w_ei * ei + w_ucb * ucb_gain

    # 4. Cost-aware: divide by cost
    scores = raw_score / cost

    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
