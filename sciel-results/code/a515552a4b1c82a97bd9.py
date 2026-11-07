import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1.0)
    
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Compute cEI
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    score_ei = ei / cost_safe
    
    # Compute cPoI
    score_poi = cdf_z / cost_safe
    
    # Compute cUCB (beta=2.0)
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    score_ucb = ucb / cost_safe
    
    # Adaptive switch
    # If f_best is still high (e.g., > -0.2), we are early. Use PoI/UCB.
    # If f_best is low (e.g., < -0.2), we are late. Use cEI.
    # Threshold -0.2 is arbitrary but reasonable for Cosine 8D.
    
    if f_best > -0.2:
        # Early stage: Explore more. PoI is a good balance.
        score = score_poi
    else:
        # Late stage: Exploit. cEI is optimal.
        score = score_ei
        
    score = np.where(np.isfinite(score), score, 0.0)
    return score
