import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we use -(mu - beta * sigma) = beta * sigma - mu
    beta = 2.0
    score = (beta * sigma - mu) / cost
    return score
