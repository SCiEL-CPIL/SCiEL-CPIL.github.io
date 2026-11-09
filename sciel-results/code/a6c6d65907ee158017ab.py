import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin
    
    # Ensure sigma is positive and not too small for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Strategy: Cost-Weighted Upper Confidence Bound (CW-UCB)
    # For minimization, we want to minimize mu - beta * sigma.
    # Maximizing -LCB is equivalent to maximizing (beta * sigma - mu).
    # We divide by cost to penalize expensive evaluations.
    
    # Beta controls the exploration-exploitation tradeoff.
    # A value around 2.0 is standard for 2D GP-UCB.
    # We can also scale beta slightly if we want, but let's start with a constant.
    beta = 2.0
    
    # Gamma controls the cost sensitivity.
    # cost = 1 + mean(x_scaled). Range [1, 2].
    # Dividing by cost^1 is a reasonable baseline.
    gamma = 1.0
    
    # Calculate UCB score
    ucb_term = beta * sigma_safe - mu
    score = ucb_term / (cost_safe ** gamma)
    
    # Check time
    if time.time() > deadline:
         # Return a simple fallback if we are running out of time
         return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
