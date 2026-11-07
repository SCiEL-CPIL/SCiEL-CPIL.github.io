import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization: Score = -mu + beta * sigma
    # Higher score means more likely to be the next best point (low value or high uncertainty)
    beta = 1.5
    return -mu + beta * sigma
