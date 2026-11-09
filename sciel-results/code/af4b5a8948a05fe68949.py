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
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # Normalized sigma for adaptive weighting
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        sigma_rel = sigma / sigma_max
    else:
        sigma_rel = np.zeros_like(sigma)
    
    # Adaptive weight: more PoI (exploration) when uncertainty is high
    # alpha in [0.5, 1.0]: high alpha = more EI (exploitation), low alpha = more PoI (exploration)
    alpha = 0.5 + 0.5 * (1.0 - sigma_rel)
    
    # Combined EI + PoI score
    combined = alpha * ei + (1.0 - alpha) * poi
    
    # Cost-aware normalization
    # Use a softer penalty (cost^1.5) to balance cost vs. improvement
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / (cost_safe ** 1.5)
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in the multimodal Shekel 4d function
    if np.max(ei) > 1e-12 and sigma_max > 1e-12:
        exploration_bonus = 0.05 * (sigma / sigma_max) * np.max(ei)
        scores += exploration_bonus / (cost_safe ** 1.5)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 320:
        return np.zeros_like(mu)
        
    return scores
