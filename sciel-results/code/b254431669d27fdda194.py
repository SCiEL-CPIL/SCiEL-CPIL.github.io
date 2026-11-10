import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization
    # Approach: Confidence Bound family, distinct from Expected Improvement
    
    # Safety clamp
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Hyperparameter for exploration-exploitation trade-off
    # Beta is typically in the range [1, 3] for standardized objectives
    beta = 1.5
    
    # UCB for minimization: We want to maximize (-mu + beta * sigma)
    # This favors low mean (exploitation) and high std dev (exploration)
    ucb_value = -mu + beta * sigma
    
    # Cost-aware score
    score = ucb_value / cost
    
    # Ensure finite scores
    # If any non-finite values appear, replace with a very low score
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
