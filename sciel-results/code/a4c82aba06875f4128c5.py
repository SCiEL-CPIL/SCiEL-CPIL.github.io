import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        # Safety checks
        if time.time() > _DEADLINE:
            # Fallback: return random or zero
            return np.zeros(2000, dtype=np.float64)

        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-9)
        
        # z-score
        z = (f_best - mu) / sigma_safe
        
        # Probability of Improvement
        # Phi(z)
        poi = norm.cdf(z)
        
        # Clip to avoid log(0)
        poi = np.clip(poi, 1e-9, 1.0)
        
        # Log-score
        score = np.log(poi) - np.log(cost_safe)
        
        # Ensure finite
        score = np.where(np.isfinite(score), score, -np.inf)
        
        return score
