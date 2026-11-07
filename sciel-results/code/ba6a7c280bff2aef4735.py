import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    We maximize the lower confidence bound of the objective function
    (minimizing the objective) normalized by the cost of evaluation.
    LCB = mu - beta * sigma.
    We want to maximize -LCB / cost = (-mu + beta * sigma) / cost.
    This balances exploitation (low mu) and exploration (high sigma).
    """
    start_time = time.time()
    deadline = 290.0 # Safety margin for 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure cost is positive
    cost_safe = np.where(cost < 1e-9, 1e-9, cost)
    
    # LCB for minimization: mu - beta * sigma
    # We want to find points with low mu and high sigma.
    # Score = - (mu - beta * sigma) / cost = (-mu + beta * sigma) / cost
    beta = 1.0
    
    scores = (-mu + beta * sigma) / cost_safe
    
    # Check deadline
    if time.time() > deadline:
         return scores
         
    return scores
