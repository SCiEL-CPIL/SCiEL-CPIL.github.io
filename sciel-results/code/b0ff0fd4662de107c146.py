import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization: maximize -mu + beta * sigma
    beta = 1.0
    return -mu + beta * sigma
