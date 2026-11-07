import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost exponent
    alpha = np.clip(2.0 - f_best, 1.0, 2.0)
    
    # Quantile-based feasibility gate:
    # Use 25th percentile of cost as the threshold.
    # Candidates below this threshold are "cheap" and get full EI.
    # Candidates above are penalized exponentially based on how much they exceed the threshold.
    cost_q25 = np.percentile(cost_safe, 25)
    
    # Scale the penalty: t = (cost - q25) / q25, clipped to [0, 10]
    t = (cost_safe - cost_q25) / np.maximum(cost_q25, 1e-9)
    t = np.clip(t, 0.0, 10.0)
    gate = np.exp(-alpha * t)
    
    scores = ei * gate
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
