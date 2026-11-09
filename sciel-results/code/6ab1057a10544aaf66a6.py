import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB with adaptive beta
    # Beta decreases as f_best decreases (we are closer to optimum)
    # f_best is the current best (lowest) value found.
    # If f_best is large, we are far, so we explore more (high beta).
    # If f_best is small, we are close, so we exploit more (low beta).
    
    # Ensure sigma is positive to avoid division issues if used, though UCB just adds.
    # But for stability, let's use a small epsilon.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta
    # Start with high beta if f_best is large.
    # f_best can be up to 25 (5^2).
    # Let's use an exponential decay of the "exploration bonus" relative to a reference.
    # Or simply linear decay?
    # beta = 10.0 * (f_best / 25.0) + 1.0 ?
    # If f_best=25, beta=11.
    # If f_best=1, beta=1.4.
    # If f_best=0.1, beta=1.04.
    
    # Let's try a slightly stronger exploration initially.
    beta = 1.0 + 10.0 * (f_best / 25.0)
    
    # UCB score: mu + beta * sigma
    # Higher is better.
    scores = mu + beta * sigma_safe
    
    # Return finite scores
    return scores
