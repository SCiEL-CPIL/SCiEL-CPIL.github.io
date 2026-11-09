"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 310.0  # Safety margin before 330s


def _check_deadline():
    """Return True if we're close to the deadline."""
    return (time.time() - _START_TIME) > _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a UCB-EI hybrid with adaptive exploration and cost-aware weighting.
    Includes a deadline guard to ensure we return within time limits.
    """
    # Deadline guard
    if _check_deadline():
        # Return a simple, fast score if we're close to timeout
        sigma_safe = np.clip(sigma, 1e-8, None)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB component
    # Adaptive beta: increase exploration as we go (more points evaluated)
    # cost is roughly 1 + mean(x scaled to [0,1]), so total_cost / 5 gives rough iteration count
    # But we don't know total_cost directly, so use a fixed reasonable beta
    beta = 0.5
    ucb = (f_best - mu) + beta * sigma_safe
    
    # Hybrid: combine EI and UCB
    # Weight EI more when we're confident (low sigma), UCB more when uncertain
    sigma_weight = sigma_safe / (sigma_safe + 0.1)  # 0 when sigma small, ~1 when large
    hybrid = (1 - sigma_weight) * ei + sigma_weight * ucb
    
    # Cost-aware weighting: prefer lower cost
    # Use a soft cost penalty rather than hard division to avoid extreme values
    cost_penalty = 1.0 / (cost + 0.5)
    score = hybrid * cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
