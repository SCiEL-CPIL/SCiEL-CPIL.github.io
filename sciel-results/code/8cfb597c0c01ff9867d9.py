import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0
_total_evals = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    Combines cost-normalized EI with adaptive UCB exploration and a
    probability-of-improvement weighting that shifts from exploration
    to exploitation as the budget is consumed.
    """
    global _start_time, _call_count, _total_evals

    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    _total_evals += 1

    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Safety fallback: simple cost-aware exploitation
        return -mu / np.maximum(cost, 1e-9)

    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-12, 1e6)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate z-score for EI
    z = (f_best - mu) / sigma

    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    cost_aware_ei = ei / cost_safe

    # Adaptive exploration: start moderate, decrease as we gather data
    # Budget is ~30 cost units; each eval costs ~1.5 on average, so ~20 evals
    # Use a smooth decay rather than step functions
    progress = min(_total_evals / 25.0, 1.0)
    exploration_factor = 0.10 * (1.0 - progress) + 0.005

    # Upper Confidence Bound (UCB) style term
    ucb_term = sigma / cost_safe * exploration_factor

    # Combine EI and exploration
    scores = cost_aware_ei + ucb_term

    # Probability of improvement (PI) for weighting
    pi = norm.cdf(z)

    # Blend: early on, trust exploration more; late, trust exploitation
    # This creates a smooth transition
    blend = 0.3 * (1.0 - progress)
    scores = scores * (1.0 + blend * pi)

    # Gentle cost penalty for expensive regions
    cost_penalty = 0.05 * np.maximum(cost_safe - 1.2, 0.0)
    scores = scores - cost_penalty

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
