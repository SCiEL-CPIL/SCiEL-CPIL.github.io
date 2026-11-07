import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0
_CALL_COUNT = 0

def acquisition(mu, sigma, f_best, cost):
    global _CALL_COUNT
    _CALL_COUNT += 1

    if (time.time() - _START_TIME) > _TIME_LIMIT:
        return -mu.copy()

    sigma = np.maximum(sigma, 1e-9)

    # Estimate remaining budget
    # Initial 5 points cost approx 1.5 each = 7.5.
    # Total budget 30.
    # Each call adds ~1.5 cost.
    # Total cost spent approx 7.5 + 1.5 * (_CALL_COUNT - 1)
    # Remaining = 30 - spent
    spent_est = 7.5 + 1.5 * max(_CALL_COUNT - 1, 0)
    remaining_budget = max(30.0 - spent_est, 0.5)
    frac_remaining = remaining_budget / 30.0

    # Adaptive Beta for UCB
    # High exploration early, low late
    beta = 1.5 + 2.5 * frac_remaining

    # Adaptive Cost Penalty Lambda
    # Low penalty early (can afford expensive exploration), High penalty late (must be cheap)
    # Lambda ranges from ~0.5 to ~2.5
    lambda_cost = 0.5 + 2.0 * (1.0 - frac_remaining)

    # UCB Score: -mu + beta * sigma
    ucb = -mu + beta * sigma

    # Cost Penalty: lambda * (cost - 1.0)
    # cost is in [1, 2]. cost-1 is in [0, 1].
    penalty = lambda_cost * (cost - 1.0)

    score = ucb - penalty

    # Safety
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    return score
