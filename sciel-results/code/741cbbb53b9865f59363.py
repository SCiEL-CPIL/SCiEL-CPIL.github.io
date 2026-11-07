import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive beta
    # f_best is in objective units.
    # We map f_best to a beta range [1.0, 5.0]
    # Linear mapping: beta = 5.0 + f_best
    # Clamp to [1.0, 5.0]
    beta = np.clip(5.0 + f_best, 1.0, 5.0)
    
    # LCB Score: Higher is better
    # We want to minimize mu - beta * sigma
    # So we maximize beta * sigma - mu
    score = beta * sigma - mu
    
    return score
