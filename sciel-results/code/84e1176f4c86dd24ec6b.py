import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Upper Confidence Bound (UCB) for minimization.
    # Score = (-mu + beta * sigma) / cost
    # This balances exploitation (low mu) and exploration (high sigma),
    # while penalizing expensive evaluations.
    
    # Beta is tuned for the small budget of this task (~20 evaluations).
    # A moderate beta encourages sufficient exploration in the initial stages.
    beta = 2.0
    
    # Calculate UCB score
    # -mu favors low predicted values (good for minimization)
    # beta * sigma favors high uncertainty (good for exploration)
    ucb_scores = -mu + beta * sigma
    
    # Divide by cost to make it cost-aware
    # cost is always >= 1.0, so this is safe
    scores = ucb_scores / cost
    
    return scores
