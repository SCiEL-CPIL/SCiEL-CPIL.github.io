import numpy as np
from scipy.special import erfc, sqrt
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement using numerically stable form
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For large positive z, Phi(z) ≈ 1, so EI ≈ f_best - mu + sigma * phi(z)
    # For large negative z, Phi(z) ≈ 0, so EI ≈ sigma * phi(z)
    # Use erfc for numerical stability: Phi(z) = 0.5 * erfc(-z/sqrt(2))
    
    # Compute Phi(z) using erfc for stability
    phi_z = np.exp(-0.5 * z * z) / sqrt(2.0 * np.pi)
    
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    # This is stable for all z
    phi_cdf = 0.5 * erfc(-z / sqrt(2.0))
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * phi_cdf + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # This encourages evaluating points that are both promising and cheap
    scores = ei / cost
    
    # Handle any non-finite values to ensure validity
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: ensure we finish well within the 300s limit
    if time.time() - start_time > 290:
        return scores
    
    return scores
