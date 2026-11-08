import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use Probability of Improvement (PI) normalized by cost
    # PI favors points that are likely to be better than f_best
    # This can help escape local minima more effectively than EI in multimodal landscapes
    z = (f_best - mu) / (sigma + 1e-6)
    pi = norm.cdf(z)
    # Add a small exploration term based on sigma to ensure we don't ignore uncertain regions
    # if they have high PI? No, PI is already low if mu is bad.
    # Let's stick to PI / cost
    return pi / cost
