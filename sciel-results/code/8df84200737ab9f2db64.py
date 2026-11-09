import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization
    # We want to minimize mu - beta * sigma
    # So we maximize beta * sigma - mu
    beta = 1.0
    scores = beta * sigma - mu
    return scores
