import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB
    # Map f_best to beta
    # f_best is the lowest observed value.
    # For sphere on [-5, 5], max value is 25.
    # We want high beta (exploration) when f_best is high (early/poor).
    # We want low beta (exploitation) when f_best is low (late/good).
    
    # Clamp f_best to [0, 25] to be safe
    f_best_clamped = np.clip(f_best, 0.0, 25.0)
    
    # Linear interpolation for beta
    beta_min = 0.5
    beta_max = 5.0
    beta = beta_min + (beta_max - beta_min) * (f_best_clamped / 25.0)
    
    # UCB score for minimization: we want to maximize -mu + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma
    score = -mu + beta * sigma
    
    return score
