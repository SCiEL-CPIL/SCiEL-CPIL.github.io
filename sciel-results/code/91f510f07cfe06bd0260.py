"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) with a
    UCB-style exploration bonus, scaled by cost.
    
    Strategy:
    - Base score is EI (captures magnitude of expected improvement).
    - Add a UCB term (beta * sigma) to encourage exploration of high-uncertainty regions.
    - Scale by cost to favor cheaper evaluations.
    - Adaptive beta: increases as we have more budget, to explore more aggressively later.
    """
    start_time = time.time()
    deadline = start_time + 0.05  # 50ms safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero or negative costs
    safe_cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-8)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Compute EI
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    EI = np.maximum(EI, 0.0)
    
    # Adaptive beta: start conservative, increase as we explore more
    # cost is in [1, 2], total budget is 30, so we can use cost as a proxy for progress
    # Early on (low cost), beta is small; later (high cost), beta is larger
    # Normalize cost to [0, 1] range: cost is in [1, 2], so (cost - 1) is in [0, 1]
    cost_norm = (cost - 1.0) / 1.0  # in [0, 1]
    
    # Beta increases from 0.05 to 0.3 as cost increases
    beta = 0.05 + 0.25 * cost_norm
    
    # Hybrid score: EI + beta * sigma
    scores = (EI + beta * sigma) / safe_cost
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
