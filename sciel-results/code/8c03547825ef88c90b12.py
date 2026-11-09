import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB parameter. 1.0 is a standard starting point.
    beta = 1.0
    
    # Calculate UCB for minimization: we want to minimize (mu - beta * sigma)
    # So the score to maximize is -(mu - beta * sigma) = -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Cost-aware: divide by cost.
    # Cost is in [1, 2].
    # This penalizes expensive evaluations.
    score = ucb_score / cost
    
    return score
