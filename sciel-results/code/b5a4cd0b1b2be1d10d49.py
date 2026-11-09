import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization: maximize -mu + beta * sigma
    # Low beta favors exploitation (low mu)
    beta = 0.2
    return -mu + beta * sigma
