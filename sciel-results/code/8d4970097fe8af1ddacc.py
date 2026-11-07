"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3D.
    
    Structural change: Additive cost-penalized Expected Improvement.
    
    Score = EI(x) - lambda * cost(x)
    
    This linear trade-off allows high-improvement candidates to overcome moderate
    cost penalties, potentially escaping local optima in cheap regions. Unlike
    multiplicative CW-EI (EI/cost), the additive penalty doesn't disproportionately
    suppress expensive regions that may hold the global minimum.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    
    # Basic safety: if anything is wrong, return a simple fallback
    if n == 0 or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(n, dtype=np.float64)
    
    # Clamp sigma to positive values for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp cost to positive values (cost is always >= 1.0 in this problem)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Additive cost penalty: EI - lambda * cost
    # Linear trade-off allows high-EI points to overcome moderate cost penalties
    lambda_cost = 0.1
    cost_penalty = lambda_cost * cost_safe
    cost_penalty = np.where(np.isfinite(cost_penalty), cost_penalty, 0.0)
    
    # Combined score: EI - lambda * cost
    # Higher score means more attractive to evaluate
    score = ei - cost_penalty
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Check time
    if time.time() > deadline - 2.0:
        # Fallback to raw EI if running out of time
        score = ei
        score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
