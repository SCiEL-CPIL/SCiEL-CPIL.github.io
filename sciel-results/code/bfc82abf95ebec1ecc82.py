import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # UCB score for minimization: S(x) = beta * sigma(x) - mu(x)
    # Higher S(x) means more promising (either low mean or high uncertainty).
    # We normalize by cost.
    
    # Parameter for exploration vs exploitation
    # 2.0 is a standard heuristic value
    beta = 2.0
    
    # Handle sigma close to 0
    # sigma is already a standard deviation, should be non-negative
    # Ensure no NaNs or Infs in input
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.0)
    
    # Calculate UCB score
    # We want to MAXIMIZE this score to select the next point
    ucb_scores = beta * sigma_safe - mu
    
    # Cost-aware: Divide by cost
    # Cost is 1 + mean(x_scaled). Min cost 1, Max cost 2.
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-8), cost, 1.0)
    
    scores = ucb_scores / cost_safe
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Replace -inf with a very small number to avoid issues if all are inf?
    # Or just 0? The problem says "Return 2000 finite scores".
    # So we must ensure finiteness.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
