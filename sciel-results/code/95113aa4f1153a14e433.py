import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    Maximizes the expected reduction in objective value per unit cost.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0
    
    # Ensure sigma is positive to avoid division by zero
    # If sigma is 0, the model is certain.
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma == 0 case separately or via epsilon
    # Using a small epsilon prevents inf/nan in z
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # z = (f_best - mu) / sigma
    # For minimization, we care about how much mu is below f_best.
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    # Vectorized operations
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    cost_safe = np.maximum(cost, eps)
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
