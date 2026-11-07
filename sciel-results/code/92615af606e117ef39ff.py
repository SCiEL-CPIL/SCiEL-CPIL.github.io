import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement per unit cost
    # This is less greedy than EI and may explore more effectively
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    return pi / cost
