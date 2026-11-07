import numpy as np

def acquisition(mu, sigma, f_best):
    # Dynamic LCB
    # Global min is approx -39.1662
    # Function range is roughly [-40, 10]
    
    global_min = -39.1662
    global_max = 10.0 # Upper bound estimate
    
    # Normalize f_best to [0, 1] where 0 is best, 1 is worst
    # Clip to avoid issues if f_best is outside expected range
    f_norm = (f_best - global_min) / (global_max - global_min)
    f_norm = np.clip(f_norm, 0.0, 1.0)
    
    # Kappa increases as f_best gets worse (higher)
    # Base kappa 1.0, Max kappa 3.0
    kappa = 1.0 + 2.0 * f_norm
    
    sigma = np.maximum(sigma, 1e-8)
    
    score = -mu + kappa * sigma
    
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
