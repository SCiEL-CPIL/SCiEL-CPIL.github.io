import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score, so we maximize the potential to find a low value.
    # Score = -mu + beta * sigma
    # A higher beta encourages exploration of uncertain regions.
    beta = 2.5
    scores = -mu + beta * sigma
    return scores
