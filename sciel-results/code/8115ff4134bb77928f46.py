import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) with robust numerical handling.
    
    Structural Change: Switches from UCB (current) / EI (inspiration) to PI.
    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    Score = PI / cost.
    
    Refinement: Deadline guard, finite checks, safe division.
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 2.0  # 2 second safety margin
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe division
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Compute Probability of Improvement
    # P(f(x) < f_best) = Phi(z)
    pi_scores = norm.cdf(z)
    
    # Cost-aware: PI per unit cost
    scores = pi_scores / cost_safe
    
    # Deadline check
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure finite output
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
