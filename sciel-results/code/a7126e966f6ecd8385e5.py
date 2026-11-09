import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # Blend EI and PoI for better exploration-exploitation balance
    # Adaptive weighting: use more PoI when uncertainty is high (sigma large)
    # This helps explore multimodal landscape better
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Normalize sigma to get relative uncertainty
    if sigma_max > 1e-9:
        sigma_rel = sigma / sigma_max
    else:
        sigma_rel = np.zeros_like(sigma)
    
    # Weight: more PoI (exploration) when uncertainty is high
    # alpha controls the blend: alpha*EI + (1-alpha)*PoI
    # When sigma is high, we want more exploration (higher weight on PoI)
    # When sigma is low, we want more exploitation (higher weight on EI)
    alpha = 0.5 + 0.5 * (1.0 - sigma_rel)  # alpha in [0.5, 1.0]
    
    # Combined score
    combined = alpha * ei + (1.0 - alpha) * poi
    
    # Cost-aware normalization
    # Cost is in [1, 2]
    # Use a softer penalty to allow some exploration of higher-cost regions
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / (cost_safe ** 1.5)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
