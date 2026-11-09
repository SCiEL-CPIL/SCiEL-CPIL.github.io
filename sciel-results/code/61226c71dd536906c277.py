import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB for minimization:
    # We want to find points where the lower bound of the confidence interval is low.
    # Lower Bound = mu - beta * sigma
    # We want to MINIMIZE this value.
    # To return a score where HIGHER is better, we can use:
    # Score = (beta * sigma - mu) / cost
    # This is the "expected improvement" of the UCB bound relative to 0? 
    # Actually, if mu is very low, -mu is very high.
    # If sigma is high, beta*sigma is high.
    # This encourages exploration (high sigma) and exploitation (low mu).
    
    beta = 2.0
    
    ucb_score = (beta * sigma - mu) / cost
    
    # Ensure finite
    ucb_score = np.nan_to_num(ucb_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    if time.time() > deadline:
        pass
        
    return ucb_score
