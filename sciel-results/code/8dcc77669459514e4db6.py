"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural Change:
    Uses a Cost-Weighted Lower Confidence Bound (LCB) approach.
    Instead of subtracting a penalty from EI, we directly compute the LCB
    (mu - kappa * sigma) which is a standard surrogate model optimization
    objective for minimization. We then scale this by a cost-aware factor.
    
    The cost-aware factor is derived from the idea that we want to prefer
    points that are both promising (low LCB) and cheap.
    
    Let's use a multiplicative penalty or a normalized score.
    
    Approach:
    1. Compute LCB: mu - kappa * sigma.
    2. We want to minimize LCB, but the acquisition function returns scores
       where higher is better. So we return -LCB.
    3. To incorporate cost, we can penalize expensive points.
       Since LCB can be negative, a simple subtraction might be unstable.
       
       Alternative: Use a "Cost-Adjusted Improvement" or similar.
       
       Let's try: Score = - (mu - kappa * sigma) / (1 + cost)
       Or: Score = - (mu - kappa * sigma) * exp(-lambda * cost)
       
       The exponential penalty is smooth and bounded.
       Cost is in [1, 2] roughly (since x is in [0,1]^4, mean is 0.5, so cost is 1.5).
       Wait, cost = 1 + mean(x_scaled). x_scaled is [0,1]. Mean is [0,1].
       So cost is in [1, 2].
       
       If we use exp(-lambda * cost), with lambda=1, cost=1 -> 0.36, cost=2 -> 0.13.
       This heavily penalizes expensive points.
       
       Let's try a simpler additive penalty on the LCB.
       Score = -(mu - kappa * sigma) - beta * cost
       
       But mu and sigma are in objective units. f_best is around 0-100 initially.
       cost is 1-2.
       The scale of LCB is similar to f_best.
       So beta * cost needs to be comparable to the LCB range.
       
       Let's look at the parent's approach:
       score = ei - beta * sigma * cost
       
       The parent used EI. EI is bounded by f_best.
       LCB is unbounded below (theoretically), but in practice bounded by data.
       
       Let's try a hybrid:
       Use LCB for exploration, but weight it by cost.
       
       Score = (f_best - (mu - kappa * sigma)) / (1 + gamma * cost)
       
       This is like "Expected Improvement divided by Cost Factor".
       This naturally balances improvement and cost.
       
       Let's refine kappa.
       Standard UCB/LCB uses kappa ~ 2-3.
       
       Let's try:
       lcb = mu - 2.0 * sigma
       improvement = f_best - lcb
       score = improvement / (1.0 + 0.5 * cost)
       
       This is a structural change from EI-based to LCB-based with multiplicative cost normalization.
    """
    # Deadline Guard
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        # Fallback: Simple LCB
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-12)
        lcb = mu - 2.0 * sigma_safe
        return -(lcb)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute LCB (Lower Confidence Bound) for minimization
    # We want low mu and high sigma (uncertainty) to be good.
    # lcb = mu - kappa * sigma
    kappa = 2.0
    lcb = mu - kappa * sigma_safe
    
    # We want to maximize the acquisition score.
    # Lower LCB is better. So we want to return -LCB.
    # But we also want to consider cost.
    
    # Cost-aware scaling:
    # We divide the "potential improvement" by a cost factor.
    # Potential improvement relative to best: f_best - lcb
    # If lcb is much lower than f_best, this is large.
    
    # Cost factor: 1 + gamma * (cost - 1)
    # Since cost is in [1, 2], cost - 1 is in [0, 1].
    # Let gamma = 1.0. Factor is in [1, 2].
    
    cost_factor = 1.0 + 1.0 * (cost - 1.0)
    
    # Score = (f_best - lcb) / cost_factor
    # This is positive when lcb < f_best.
    
    score = (f_best - lcb) / cost_factor
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
