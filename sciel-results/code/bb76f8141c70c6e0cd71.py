import numpy as np

def acquisition(mu, sigma, f_best):
    beta = 2.0
    score = -mu + beta * sigma
    return score
