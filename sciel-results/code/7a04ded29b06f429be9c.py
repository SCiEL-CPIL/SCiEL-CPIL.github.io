import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Add a small exploration parameter xi
    xi = 0.01
    z = (f_best - mu - xi) / (sigma + 1e-9)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    return ei / (cost + 1e-9)
