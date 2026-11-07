import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-8)
    
    # Adaptive Beta
    # If f_best is large, we are far from optimum, explore more.
    # If f_best is small, we are close, exploit more.
    # Scale f_best to [0, 1] roughly.
    # Beale min is 0. Max in domain is ~ 100-200.
    # Let's use a sigmoid or simple ratio.
    # beta = 1.0 + 4.0 * (f_best / (f_best + 5.0))
    # If f_best = 0, beta = 1.0
    # If f_best = 5, beta = 3.0
    # If f_best = 100, beta ~ 5.0
    
    beta = 1.0 + 4.0 * (f_best / (f_best + 5.0))
    
    score = -mu + beta * sigma
    
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    return score
