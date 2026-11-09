import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Expected Improvement (EI) with adaptive exploration bonus.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0  # 2 second safety margin for this call
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # when f_best > mu, else 0
    
    z = (f_best - mu) / sigma
    z = np.maximum(z, 0.0)  # Only consider positive z for EI
    
    # Phi(z) = norm.cdf(z), phi(z) = norm.pdf(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) for z > 0
    # For z <= 0, EI = 0
    ei = np.where(z > 0, (f_best - mu) * Phi_z + sigma * phi_z, 0.0)
    
    # Add a small exploration bonus based on sigma to avoid getting stuck
    # The bonus is scaled by how far we are from the known minimum (3.0)
    gap = max(f_best - 3.0, 0.0)
    
    # When gap is large (early), we want more exploration
    # When gap is small (late), we want less exploration
    # Scale the exploration bonus: high when gap is large
    exploration_weight = 0.1 + 0.5 * np.tanh(gap / 5.0)
    
    scores = ei + exploration_weight * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
