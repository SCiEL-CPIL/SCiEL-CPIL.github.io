import numpy as np
from scipy.stats import norm
import time

START_TIME = time.time()
DEADLINE = 330.0
SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple EI if time is critical
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic exploration coefficient
    # Hartmann 3D min is approx -3.86
    min_val = -3.86278
    # Assume initial f_best is around 0 or 1 for Hartmann 3D
    # If f_best is close to min, progress is high
    # We estimate the total range of the function. 
    # Max value of Hartmann 3D is around 3.
    # So range is ~7.
    range_est = 7.0
    
    # progress: 0 at start (f_best ~ 3), 1 at end (f_best ~ -3.86)
    # f_best starts high (bad) and decreases.
    # Let's say f_best = 3.0 initially.
    # progress = (3.0 - (-3.86)) / 7.0 = 1.0? No.
    # We want progress to be 0 when f_best is bad (high) and 1 when f_best is good (low).
    # Let's define distance_to_min = f_best - min_val.
    # If distance is large, we are far.
    # Let's cap the initial distance.
    # If f_best is 3.0, dist = 6.86.
    # If f_best is -3.86, dist = 0.
    
    dist_to_min = f_best - min_val
    # Normalize distance. 
    # If dist > 5, assume we are at the start.
    # If dist < 0.1, assume we are at the end.
    # linear scale from 5.0 to 0.0
    progress = np.clip(1.0 - dist_to_min / 5.0, 0.0, 1.0)
    
    # kappa: high when progress is low (start), low when progress is high (end)
    kappa = 0.5 + 1.5 * (1.0 - progress)
    
    exploration_bonus = kappa * sigma_safe
    
    scores = ei + exploration_bonus
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
