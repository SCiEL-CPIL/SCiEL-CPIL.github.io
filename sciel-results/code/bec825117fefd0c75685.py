import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Calculate z-score
    # Avoid division by zero
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Adaptive beta based on the scale of the objective function
    # Estimate scale from the spread of mu
    # If mu is constant, scale is 0, beta is 1.
    # If mu is spread out, beta is smaller.
    scale = np.std(mu)
    beta = 1.0 / (scale + 1.0)
    
    # Multiplicative form: PI * (1 + beta * sigma)
    # This couples exploitation (PI) and exploration (sigma)
    # Dividing by cost makes it cost-aware
    scores = (pi * (1.0 + beta * sigma)) / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
