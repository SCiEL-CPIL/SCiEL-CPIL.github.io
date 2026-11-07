import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Adaptive beta
    # f_best is a scalar (lowest observed value)
    # We want high beta when f_best is large (far from min)
    # Low beta when f_best is small (close to min)
    
    # Clip f_best to avoid issues if it's negative (shouldn't happen for Ackley)
    fb = max(0.0, f_best)
    
    # Linear decay from 4.0 at fb=5 to 1.0 at fb=0
    beta = 1.0 + 3.0 * (fb / 5.0)
    beta = np.clip(beta, 1.0, 4.0)
    
    # LCB score: maximize (beta * sigma - mu)
    # Normalize by cost
    # Ensure sigma is positive and non-zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    score = (beta * sigma_safe - mu) / cost
    
    # Return finite scores
    # If cost is 0 (shouldn't happen, min cost 1), handle it
    score = np.where(cost > 1e-6, score, 0.0)
    
    return score
