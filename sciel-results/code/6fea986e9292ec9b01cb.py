import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # LCB (Lower Confidence Bound) for minimization
    # We want to minimize f.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to select x that minimizes LCB(x).
    # Therefore, we want to maximize -LCB(x) = -mu(x) + beta * sigma(x).
    # To normalize and relate to f_best, we can use:
    # Score = (f_best - LCB(x)) / Cost(x)
    # This is positive if LCB(x) < f_best, indicating potential for improvement.
    # It is larger for higher uncertainty (beta * sigma) and lower mean (mu).
    
    # Adaptive beta: 
    # In early stages, we want more exploration (higher beta).
    # In later stages, we want more exploitation (lower beta).
    # We don't have iteration count, but we can infer from f_best?
    # Or just use a fixed high beta.
    # Let's try beta = 2.5.
    
    beta = 2.5
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    lcb = mu - beta * sigma_safe
    
    # Score: how much better is the lower bound than the current best, normalized by cost
    # If lcb is much lower than f_best, score is high.
    score = (f_best - lcb) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
