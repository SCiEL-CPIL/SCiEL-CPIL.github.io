import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB (Upper Confidence Bound) for minimization
    # We aim to minimize f(x).
    # The Lower Confidence Bound (LCB) is mu - kappa * sigma.
    # We want to select the point with the lowest LCB (most promising minimum).
    # To create a score where higher is better, we negate the LCB:
    # Score = -(mu - kappa * sigma) = kappa * sigma - mu
    # We divide by cost to account for the cost-aware requirement.
    
    kappa = 2.0
    
    # Ensure cost is not zero to avoid division by zero
    cost_safe = np.where(cost < 1e-10, 1e-10, cost)
    
    scores = (kappa * sigma - mu) / cost_safe
    
    return scores
