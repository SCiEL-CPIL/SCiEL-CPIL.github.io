import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    epsilon = 1e-9
    z = (f_best - mu) / (sigma + epsilon)
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return scores
