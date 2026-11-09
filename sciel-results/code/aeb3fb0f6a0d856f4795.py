import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization
    # We want to maximize the score.
    # Low mu is good. High sigma is good.
    # Score = -mu + beta * sigma
    # beta = 1.5 is a reasonable balance for 2D Ackley
    beta = 1.5
    return -mu + beta * sigma
