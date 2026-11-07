import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB acquisition function
    # We want to maximize (f_best - mu + beta * sigma) / cost
    # Higher score means more likely to improve and worth the cost.
    
    beta = 1.5  # Exploration-exploitation trade-off
    
    # Calculate the numerator
    # f_best - mu is the expected improvement if we ignore uncertainty
    # beta * sigma is the exploration bonus
    numerator = f_best - mu + beta * sigma
    
    # Divide by cost
    # cost is > 0 (min 1.0)
    scores = numerator / cost
    
    # Ensure finite scores
    # If cost is 0 (shouldn't happen), replace with 1
    # If numerator is inf, replace
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
