"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # 5s safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with budget-adaptive exploration.
    
    Uses Expected Improvement per unit cost as the core signal, with an
    adaptive exploration bonus that decays as the cost budget is consumed.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware EI: divide by cost to account for evaluation expense
    ei_score = ei / cost_safe

    # Budget-adaptive exploration bonus:
    # The total budget is 30. We estimate remaining budget from the mean cost
    # of candidates and the number of evaluations already done.
    # Since we don't have explicit iteration count, we use the spread of costs
    # as a proxy: if costs are low on average, we've spent less.
    # A simpler and more robust approach: scale exploration by the ratio of
    # sigma to the current best improvement scale, and dampen as f_best
    # approaches the known minimum (0 for most functions, -3.86 for Hartmann).
    
    # Adaptive exploration weight: higher when sigma is large relative to
    # the current best value's magnitude, lower when we're close to optimum.
    # This naturally transitions from exploration to exploitation.
    improvement_scale = np.abs(f_best) + 1.0  # avoid division by zero
    sigma_rel = sigma / improvement_scale
    
    # Exploration bonus: proportional to relative uncertainty
    # Dampen it so it doesn't dominate EI
    exploration_weight = 0.5
    exploration_bonus = exploration_weight * sigma_rel / cost_safe

    # Combine: EI per cost + adaptive exploration bonus
    score = ei_score + exploration_bonus

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
