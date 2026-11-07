import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Probability of Improvement (PoI) with a Gaussian exploration bonus,
    both normalized by cost. This balances the robustness of PoI (which focuses on
    the likelihood of improvement) with the exploration capability of an uncertainty
    term (which helps escape local optima in multimodal functions like Rastrigin).
    """
    # Deadline guard: if we're running out of time, return a simple score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return (f_best - mu) / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)

    # 1. Probability of Improvement (PoI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)

    # 2. Exploration Bonus
    # Use a Gaussian of sigma to emphasize high uncertainty regions.
    # This is smoother than raw sigma and less aggressive than LCB.
    # We scale it to be comparable to PoI (which is in [0, 1]).
    # The standard deviation of the Gaussian determines the "reach" of exploration.
    # A value of 0.5 is a reasonable heuristic for 2D problems scaled to [0,1] with length scale 0.2.
    exploration_width = 0.5
    exploration_bonus = np.exp(-(sigma_safe / exploration_width) ** 2)
    
    # Weight the exploration bonus. 
    # PoI is the primary driver, but we add a significant exploration component.
    # A weight of 0.5 suggests that when PoI is low, we still have a strong drive to explore.
    exploration_weight = 0.5
    weighted_exploration = exploration_weight * exploration_bonus

    # 3. Combine and Normalize by Cost
    # The total "value" of a point is the weighted sum of exploitation (PoI) and exploration.
    # We divide by cost to account for the expensive evaluations.
    # Cost is in [1, 2]. Dividing by cost penalizes expensive points.
    raw_score = poi + weighted_exploration
    score = raw_score / cost_safe

    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
