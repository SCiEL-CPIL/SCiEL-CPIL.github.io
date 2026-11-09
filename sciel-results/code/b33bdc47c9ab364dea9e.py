import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # We want to find x that minimizes f(x).
    # UCB(x) = mu(x) - beta * sigma(x) is the upper bound of the function value.
    # We want to MINIMIZE this upper bound.
    # So we want to MAXIMIZE (beta * sigma(x) - mu(x)).
    # We also divide by cost to prefer cheaper points.
    
    beta = 2.5
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB score component: higher is better (more exploration + better mean)
    ucb_component = beta * sigma - mu
    
    # Cost-aware score
    score = ucb_component / cost_safe
    
    return score
