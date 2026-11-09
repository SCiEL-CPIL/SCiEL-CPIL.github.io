import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline guard
_global_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to leave before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s wall-clock limit, return quickly
    elapsed = time.time() - _global_start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a safe fallback: uniform random scores to avoid invalid output
        n = len(mu) if hasattr(mu, '__len__') else 2000
        return np.random.uniform(0, 1, size=n)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = 2000
    if mu.shape != (n,):
        mu = mu.reshape(-1)
        if mu.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if sigma.shape != (n,):
        sigma = sigma.reshape(-1)
        if sigma.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if cost.shape != (n,):
        cost = cost.reshape(-1)
        if cost.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Use a different acquisition function family.
    # Instead of pure EI or EI/cost, we use a combination of:
    # 1. Expected Improvement (exploitation)
    # 2. Uncertainty bonus (exploration)
    # 3. Cost-awareness (efficiency)
    
    # Normalize EI to [0, 1] range for better combination with other terms
    ei_max = np.max(ei)
    if ei_max > eps:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Uncertainty bonus: prioritize areas with high sigma
    # This helps explore unexplored regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = sigma_safe
    
    # Cost-awareness: prefer lower cost points
    # Normalize cost to [0, 1] where 1 is lowest cost
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    if cost_max - cost_min > eps:
        cost_norm = (cost_max - cost_safe) / (cost_max - cost_min)  # 1 = lowest cost
    else:
        cost_norm = np.ones_like(cost_safe)
    
    # Combine terms with weights
    # Weighting: 0.6 for exploitation (EI), 0.2 for exploration (sigma), 0.2 for efficiency (cost)
    # This balances finding the minimum while exploring and being cost-effective
    w_ei = 0.6
    w_sigma = 0.2
    w_cost = 0.2
    
    scores = w_ei * ei_norm + w_sigma * sigma_norm + w_cost * cost_norm
    
    # Check for non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf, return zeros to avoid invalid results
    if not np.any(np.isfinite(scores)):
        return np.zeros(n, dtype=np.float64)
    
    return scores
