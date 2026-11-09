import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB (Upper Confidence Bound) for minimization
    # Higher score means more likely to be evaluated
    # We want low mu and high sigma
    # Cost is in the denominator to prefer cheaper points
    beta = 1.5
    scores = (-mu + beta * sigma) / cost
    return scores
