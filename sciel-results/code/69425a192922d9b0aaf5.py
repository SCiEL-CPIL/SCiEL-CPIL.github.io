import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Exploration parameter
    beta = 3.0
    
    # UCB score for minimization
    # Higher score is better
    # -mu encourages low predicted values
    # beta * sigma encourages high uncertainty (diversity/exploration)
    ucb_scores = -mu + beta * sigma
    
    # Normalize by cost to prefer cheaper evaluations
    # cost is at least 1, so division is safe
    scores = ucb_scores / cost
    
    return scores
