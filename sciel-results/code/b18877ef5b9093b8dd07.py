"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI components
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (minimization perspective)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Fallback to simple EI/cost
        scores = ei / cost
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Batch-aware diversity promotion strategy
    # We approximate the diversity penalty by using the uncertainty (sigma) 
    # as a proxy for "distance" from known data points, but scaled appropriately.
    # A more sophisticated approach would require coordinates, which are not provided.
    # Instead, we use a combination of EI and a bonus for high uncertainty (exploration)
    # that is dampened by the cost, encouraging exploration in high-uncertainty, 
    # low-cost regions.
    
    # Normalize EI to [0, 1] to make it comparable with other terms
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_norm = (ei - ei_min) / ei_range
    
    # Use sigma as a proxy for exploration potential, but normalize it
    sigma_max = np.max(sigma) if np.max(sigma) > 0 else 1.0
    sigma_min = np.min(sigma)
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Weighted combination: 70% exploitation (EI), 30% exploration (sigma)
    # The cost penalty is applied to the exploration term to avoid wasting budget
    # on high-cost, high-uncertainty points
    exploration_bonus = sigma_norm / (1.0 + cost)
    scores = 0.7 * ei_norm + 0.3 * exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
