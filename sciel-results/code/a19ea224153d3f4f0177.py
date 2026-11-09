import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0
_total_evals = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    Combines the structural idea of cost-aware EI (from inspiration 1) with
    refinement machinery: adaptive UCB blend, probability-of-improvement boost,
    and cost penalty (from the current Thompson sampling program's exploration
    bonus and the inspirations' cost awareness).
    """
    global _start_time, _call_count, _total_evals

    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    _total_evals += 1

    elapsed = time.time() - _start_time
    if elapsed > 315:
        # Safety fallback: simple cost-aware exploitation
        return -mu / np.maximum(cost, 1e-9)

    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-12, 1e6)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    cost_aware_ei = ei / cost_safe

    # Adaptive UCB exploration term
    # In early iterations, explore more; in later iterations, exploit more
    if _total_evals <= 10:
        ucb_beta = 0.15
    elif _total_evals <= 25:
        ucb_beta = 0.10
    elif _total_evals <= 50:
        ucb_beta = 0.05
    else:
        ucb_beta = 0.02

    # UCB score: -mu + beta * sigma, divided by cost
    ucb_score = (-mu + ucb_beta * sigma) / cost_safe

    # Blend EI and UCB: primary is cost-aware EI, with UCB for exploration
    # In early iterations, give more weight to UCB
    if _total_evals <= 10:
        alpha = 0.3
    elif _total_evals <= 25:
        alpha = 0.2
    elif _total_evals <= 50:
        alpha = 0.1
    else:
        alpha = 0.05

    scores = (1.0 - alpha) * cost_aware_ei + alpha * ucb_score

    # Probability of improvement boost
    pi = norm.cdf(z)
    scores = scores * (1.0 + 0.15 * pi)

    # Cost penalty for very expensive evaluations
    cost_penalty = np.where(cost_safe > 1.5, 0.05 * (cost_safe - 1.5), 0.0)
    scores = scores - cost_penalty

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
