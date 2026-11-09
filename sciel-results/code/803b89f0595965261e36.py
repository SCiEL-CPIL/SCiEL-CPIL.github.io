import numpy as np

def acquisition(mu, sigma, f_best, cost):
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # UCB for minimization
    # We want to find the point with the lowest possible value.
    # The lower bound of the posterior is mu - beta * sigma.
    # We want to maximize the "potential" for improvement, which is related to - (mu - beta * sigma) = -mu + beta * sigma.
    # Using a higher beta encourages exploration of high uncertainty regions.
    beta = 2.0
    
    ucb_term = -mu + beta * sigma_safe
    
    # Cost-aware gating
    # Exponential penalty for higher cost
    gate = np.exp(-0.5 * cost_safe)
    
    score = ucb_term * gate
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
