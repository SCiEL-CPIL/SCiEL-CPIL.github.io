import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # We want to maximize the score, so we return - (mu - beta * sigma)
    # which is -mu + beta * sigma
    beta = 2.0
    return -mu + beta * sigma
