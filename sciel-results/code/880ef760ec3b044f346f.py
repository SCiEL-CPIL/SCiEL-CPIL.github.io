"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE_START = time.time()
_DEADLINE_LIMIT = 320.0

def _remaining_time():
    return max(0.0, _DEADLINE_LIMIT - (time.time() - _DEADLINE_START))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of Expected Improvement (EI)
    and Uncertainty (UCB-like) terms, both normalized by evaluation cost.
    
    Parameters:
        mu: (2000,) GP mean in objective units
        sigma: (2000,) GP std in objective units
        f_best: scalar, lowest observed value
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) scores, higher = evaluate next
    """
    # Deadline guard
    if _remaining_time() < 2.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # 1. Expected Improvement (EI)
    # EI is 0 if mu >= f_best, else positive.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost_safe

    # 2. Uncertainty / Exploration Term
    # We reward high uncertainty (sigma) per unit cost.
    # This drives the search to unexplored regions, preferring cheaper ones.
    # The weight beta controls the exploration/exploitation trade-off.
    # A higher beta encourages more exploration.
    # Given the Levy function's ruggedness and the cost bias towards the origin,
    # we need sufficient exploration to find the shifted minimum.
    beta = 1.5 
    score_explore = beta * sigma_safe / cost_safe

    # 3. Combine
    # The additive combination is standard for UCB-like methods.
    # EI handles exploitation (improvement over best).
    # Sigma handles exploration (potential for much better values).
    scores = score_ei + score_explore

    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
