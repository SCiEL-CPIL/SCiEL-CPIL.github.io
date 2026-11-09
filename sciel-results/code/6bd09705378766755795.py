import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization
    # Maximize -mu + beta * sigma
    # Cost-aware: divide by cost
    
    # Numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Exploration parameter
    # Tuned for small budget (approx 25 evals)
    beta = 2.5
    
    # Upper Confidence Bound for minimization
    # We want to find low f(x), so we prefer low mu.
    # -mu is high when mu is low.
    ucb = -mu + beta * sigma
    
    # Cost-aware score
    scores = ucb / cost
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
