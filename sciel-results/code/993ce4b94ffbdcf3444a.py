"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time guard
_start_time = None
_DEADLINE = 300.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses a combination of Expected Improvement (EI) and Uncertainty (UCB)
    with exponential cost gating to balance exploration and exploitation
    while favoring cheaper evaluations.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety margin: return simple uncertainty-based score
        sigma_safe = np.maximum(sigma, 1e-10)
        return sigma_safe / cost

    # Ensure sigma is safe
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB component (uncertainty-based exploration)
    # beta controls exploration-exploitation tradeoff
    beta = 2.0
    ucb = mu + beta * sigma_safe
    # For minimization, lower UCB is better, so we negate for "higher = better"
    ucb_score = -ucb

    # Cost-aware gating: exponential penalty for higher cost
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    lambda_cost = 1.5
    cost_gate = np.exp(-lambda_cost * (cost - 1.0))

    # Combine EI and UCB with cost gating
    # EI is zero in most regions, UCB is non-zero everywhere
    # Weight them to get good exploration in high dimensions
    ei_weight = 1.0
    ucb_weight = 0.5

    score = cost_gate * (ei_weight * ei + ucb_weight * ucb_score)

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
