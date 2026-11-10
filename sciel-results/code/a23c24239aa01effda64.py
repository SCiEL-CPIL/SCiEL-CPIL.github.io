import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # We aim to minimize the function, so we look for points with low
    # Lower Confidence Bound: LCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the "improvement" potential, which is -LCB(x).
    # Score = -LCB(x) / cost(x) = (beta * sigma(x) - mu(x)) / cost(x)
    
    # Safety clamp for sigma and cost to avoid division by zero or negative variance
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Exploration parameter
    # A value of 1.5 provides a balance between exploration and exploitation
    # for typical Bayesian Optimization tasks.
    beta = 1.5
    
    # Calculate the score
    # Note: mu and sigma are in objective units
    score = (beta * sigma - mu) / cost
    
    # Handle numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
