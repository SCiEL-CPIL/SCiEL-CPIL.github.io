import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: maximize (beta * sigma - mu) / cost
    # beta controls the exploration-exploitation trade-off
    beta = 3.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate UCB score
    # We want to find points that are likely to be lower than f_best.
    # Optimistic estimate of f(x) is mu - beta * sigma.
    # Improvement is f_best - (mu - beta * sigma) = f_best - mu + beta * sigma.
    # Since f_best is constant, ranking is determined by (beta * sigma - mu).
    # We divide by cost to prefer cheaper points.
    
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
