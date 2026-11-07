"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
TIME_LIMIT = 290.0 # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check time
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback: return cost-aware EI
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI: maximize EI per unit cost
    # This is the standard approach for cost-aware BO
    cai = ei / cost
    
    # Also consider a cost-aware LCB
    # Adaptive beta: larger beta when we have less information
    # Use a simple heuristic: beta based on the relative uncertainty
    beta = 1.5 + 0.5 * np.exp(-f_best / (1.0 + abs(f_best)))
    lcb = mu + beta * sigma_safe
    # Score for LCB: we want low LCB, so score = -lcb / cost
    lcb_score = -lcb / cost
    
    # Combine: use a weighted combination
    # Weight toward CAI when f_best is close to 0 (exploit)
    # Weight toward LCB when f_best is far from 0 (explore)
    # f_best is the minimum value, so if it's close to 0, we're doing well
    weight_cai = 0.7 + 0.3 * np.exp(-abs(f_best))
    weight_lcb = 1.0 - weight_cai
    
    score = weight_cai * cai + weight_lcb * lcb_score
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
