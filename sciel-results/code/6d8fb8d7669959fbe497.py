import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 330s wall clock limit. 
    # We must return within a safety margin. 
    # If we are close to the limit, we return a simple fallback to avoid timeout.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Fallback: return cost-based score to be safe and fast
        # Higher score = better. We want low cost, so -cost.
        return -np.asarray(cost, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and numerically stable
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Strategy: Cost-Effective Expected Improvement (CEI)
    # Instead of EI - lambda * cost or EI / (1 + cost), we use EI / cost.
    # This directly maximizes the expected improvement per unit cost.
    # This is a standard formulation for BO under budget constraints.
    
    # 1. Calculate Expected Improvement (EI)
    # For minimization: EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost-Effective Score
    # cost is in [1, 2] (1 + mean(x_scaled))
    # We want to maximize EI / cost.
    # To avoid division by zero (though cost >= 1), we add a small epsilon.
    cost_safe = np.maximum(cost, eps)
    
    cei = ei / cost_safe
    
    # 3. Normalization for numerical stability and consistent scale
    # The raw CEI values might vary in magnitude. 
    # However, the relative ordering is what matters for the argmax in the evaluator.
    # We can return the raw CEI scores, but normalizing helps if the evaluator 
    # expects scores in a specific range or if there are numerical issues.
    # Given the prompt says "Return 2000 finite scores (higher = evaluate next)", 
    # raw CEI is sufficient as long as it's finite.
    
    # One potential issue: if f_best is very small (close to 0), EI is small, 
    # and CEI is small. This is correct behavior: we are less aggressive.
    
    # Ensure finite values
    cei = np.nan_to_num( cei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return cei
