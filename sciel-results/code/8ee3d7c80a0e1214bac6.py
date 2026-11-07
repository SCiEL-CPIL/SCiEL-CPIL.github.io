import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    timeout = 1.0 # 1 second limit per call, generous for 2000 points
    
    try:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        
        # Cost efficiency: 1 / cost
        # cost is in [1, 2]
        cost_eff = 1.0 / np.maximum(cost, 1e-9)
        
        scores = poi * cost_eff
        
        # Check time
        if time.time() - start_time > timeout:
             # Fallback: just return POI if it took too long (unlikely)
             return poi
        
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores
    except Exception:
        # Fallback to simple PI if error
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        return np.nan_to_num(poi, nan=0.0, posinf=0.0, neginf=0.0)
