import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 290.0  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Return a fallback, e.g., pure exploitation or random
        # Since we need to return finite scores, we can return -mu/cost (greedy)
        return -mu / np.maximum(cost, 1e-9)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB for minimization: we want to minimize f.
    # Optimistic value is mu - beta * sigma.
    # We want to maximize the "desirability".
    # Desirability is high when optimistic value is low.
    # So score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    beta = 2.0
    score = (beta * sigma - mu) / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
