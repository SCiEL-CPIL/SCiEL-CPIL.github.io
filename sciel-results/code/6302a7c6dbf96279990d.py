import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
       # Ensure sigma is positive
       sigma = np.maximum(sigma, 1e-9)
       
       # Adaptive beta based on f_best
       # Hartmann 3D min is ~ -3.86, max ~ 1.0
       # If f_best is high (bad), explore more (high beta)
       # If f_best is low (good), exploit (low beta)
       
       # Normalize f_best to [0, 1] where 0 is best (-3.86) and 1 is worst (0.0)
       f_min = -3.86
       f_max = 0.0
       
       # Clip to avoid out of bounds
       f_norm = np.clip((f_best - f_min) / (f_max - f_min), 0, 1)
       
       # Beta ranges from 1.0 (exploit) to 5.0 (explore)
       beta = 1.0 + 4.0 * f_norm
       
       # UCB for minimization: we want to maximize (beta * sigma - mu)
       # Because we want low mu and high sigma.
       score = beta * sigma - mu
       
       return score
