import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # LCB: Lower Confidence Bound for minimization
    # We want to minimize mu - beta * sigma
    # Acquisition score should be high for good candidates
    # So score = - (mu - beta * sigma) / cost
    beta = 2.0
    lcb = mu - beta * sigma
    score = -lcb / cost
    return score
