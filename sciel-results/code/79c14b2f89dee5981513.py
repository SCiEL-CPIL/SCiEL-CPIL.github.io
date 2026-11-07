import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Time guard
    # ...
    
    # UCB for minimization
    kappa = 2.5
    ucb = -mu + kappa * sigma
    
    # Cost aware
    scores = ucb / cost
    
    return scores
