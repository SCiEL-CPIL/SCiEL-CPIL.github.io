"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses a cost-normalized Expected Improvement (EI) acquisition function.
    This is a structural change from the previous LCB approach, utilizing
    probabilistic improvement estimates normalized by evaluation cost.
    
    Args:
        mu: Array of GP means (shape 2000,)
        sigma: Array of GP standard deviations (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Array of evaluation costs (shape 2000,)
        
    Returns:
        Array of acquisition scores (shape 2000,), higher is better.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Convert to numpy arrays and ensure float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    # Replace non-finite mu with f_best (neutral)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    # Replace non-finite or tiny sigma with a small positive value
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    # Replace non-finite or tiny cost with 1.0 (minimum cost)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Calculate Expected Improvement (EI) for minimization
    # Improvement I = max(0, f_best - Y)
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Calculate z-scores
    # Avoid division by zero (sigma_safe is > 1e-9, so this is safe, but clip for stability)
    z = (f_best - mu_safe) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in CDF/PDF
    # norm.cdf handles large values well, but clipping ensures stability
    z = np.clip(z, -10, 10)
    
    # Calculate PDF and CDF
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Calculate EI
    # If sigma is very small, z is large. 
    # If mu < f_best, z > 0, cdf ~ 1, pdf ~ 0. EI ~ f_best - mu.
    # If mu > f_best, z < 0, cdf ~ 0, pdf ~ 0. EI ~ 0.
    ei = (f_best - mu_safe) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Maximize EI per unit cost
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline:
         return score

    return score
# EVOLVE-BLOCK-END
