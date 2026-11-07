# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Lower Confidence Bound (LCB).
    This approach balances exploitation (low mean) and exploration (high uncertainty)
    using a confidence bound, which is distinct from Improvement-based metrics.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Beta controls the exploration-exploitation tradeoff.
    # A higher beta encourages more exploration.
    # We use a fixed beta of 2.0, which is a common heuristic.
    beta = 2.0
    
    # LCB: mu - beta * sigma
    # We want to minimize this, so we maximize its negative.
    lcb = mu - beta * sigma_safe
    
    # Score is -LCB / cost
    score = -lcb / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
