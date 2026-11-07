import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement with a small exploration constant
    xi = 0.1
    f_best_adj = f_best - xi
    z = (f_best_adj - mu) / (sigma + 1e-9)
    return (f_best_adj - mu) * norm.cdf(z) + sigma * norm.pdf(z)
