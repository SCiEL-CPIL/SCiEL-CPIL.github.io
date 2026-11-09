# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a log-space multiplicative utility model.
    
    Uses the formula: log(max(f_best - (mu - kappa*sigma), eps)) - beta * log(cost)
    This models optimization as multiplicative utility maximization rather than
    linear ratio, which prevents aggressive bias toward low-cost regions early on.
    
    Higher scores indicate points that are more likely to improve f_best
    relative to their evaluation cost.
    """
    eps = 1e-12
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid log of negative or zero
    sigma_safe = np.maximum(sigma, eps)
    
    # LCB with exploration parameter
    # kappa=1.5 based on knowledge that fixed kappa=1.5 outperformed adaptive
    kappa = 1.5
    lcb = mu - kappa * sigma_safe
    
    # We want to maximize the improvement over f_best in log space
    # improvement = f_best - lcb (lower lcb means more potential improvement)
    # But we need to be careful: if lcb > f_best, improvement is negative
    # We use max(f_best - lcb, eps) to keep log well-defined and favor points
    # where lcb is well below f_best
    
    # Actually, let's think about this more carefully.
    # The goal is to find points where the predicted minimum (lcb) is as low as possible,
    # but penalize by cost.
    # 
    # In log-space multiplicative model:
    # We want to maximize: (potential_improvement) / (cost^beta)
    # Taking log: log(potential_improvement) - beta * log(cost)
    #
    # potential_improvement should be how much lower lcb is than f_best
    # But if lcb is already below f_best, that's good.
    # 
    # Let's use: improvement = f_best - lcb
    # If improvement > 0, the point has potential to beat f_best
    # We want to maximize improvement / cost^beta
    # In log space: log(improvement) - beta * log(cost)
    #
    # But if improvement <= 0, we should give low score.
    # We can use log(max(improvement, eps)) which will be very negative for small improvement.
    
    improvement = f_best - lcb
    
    # For points where improvement is very small or negative, log will be very negative
    # This naturally deprioritizes them
    log_improvement = np.log(np.maximum(improvement, eps))
    
    # Cost penalty in log space
    # cost is in range [1, 2] (since 1 + mean(x_scaled) where x_scaled in [0,1])
    # beta controls how strongly we penalize cost
    # beta=0.5 means we penalize sqrt(cost), which is moderate
    beta = 0.5
    log_cost_penalty = beta * np.log(cost + eps)
    
    # Score: higher is better
    score = log_improvement - log_cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
