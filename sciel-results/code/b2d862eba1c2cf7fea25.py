"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return simple cost-based scores if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 280:  # Safety margin before 300s limit
        # Fallback: prefer low cost, high uncertainty
        return sigma / (cost + 1e-9)
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost efficiency: EI per unit cost
    ei_cost = ei / cost
    
    # Exploration bonus: reward high uncertainty (sigma) relative to cost
    # This helps escape local minima in the multimodal function
    exploration = 0.5 * sigma / cost
    
    # Combined score: balance exploitation (EI/cost) with exploration
    score = ei_cost + exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
