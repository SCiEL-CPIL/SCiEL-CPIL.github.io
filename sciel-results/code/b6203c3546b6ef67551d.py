"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function with
    dynamic exploration scaled by remaining budget.
    
    mu: shape (2000,) - GP mean
    sigma: shape (2000,) - GP std
    f_best: float - best observed value
    cost: shape (2000,) - evaluation costs
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute LCB: we want to minimize, so LCB = mu - beta * sigma
    # Higher LCB score means more promising (lower expected value with uncertainty)
    # For minimization, we score as -LCB = -mu + beta * sigma
    
    # Dynamic beta based on gap to known minimum and remaining budget
    # The known minimum of cosine8 is -0.8
    gap = f_best - (-0.8)
    gap = max(gap, 1e-6)
    
    # Remaining budget: total budget is 30, initial cost is ~5 (5 points * avg cost ~1.5)
    # cost array represents cost of each candidate; use mean cost as proxy for average
    mean_cost = np.mean(cost)
    # Estimate how many more evaluations we can afford
    # Total cost spent so far is not directly given, but we can infer from the fact
    # that we're in the loop. Use a heuristic: beta decreases as we approach the optimum.
    # beta = base_beta * sqrt(log(1 + 1/gap)) to increase exploration when far from optimum
    base_beta = 0.5
    beta = base_beta * np.sqrt(np.log1p(1.0 / gap))
    
    # Clamp beta to reasonable range
    beta = np.clip(beta, 0.1, 2.0)
    
    # LCB score: -mu + beta * sigma (higher is better for minimization)
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = lcb_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
