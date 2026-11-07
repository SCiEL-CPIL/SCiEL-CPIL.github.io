import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) for Bayesian optimization.
    
    Strategy:
    - Use standard EI for minimization, which naturally balances exploration and exploitation.
    - Apply a soft cost penalty using square root of cost, which is less aggressive than 
      linear cost division, allowing the search to explore higher-cost regions if they 
      offer significantly higher expected improvement.
    - Include numerical stability safeguards for finite scores.
    """
    # Deadline guard: if we're getting close to the time limit, fall back to a simple heuristic
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Simple fallback: prioritize low mu with mild cost penalty
        scores = -mu / np.sqrt(np.maximum(cost, 1e-8))
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in CDF/PDF calculations
    # For large positive z, EI approaches f_best - mu
    # For large negative z, EI approaches 0
    z_clipped = np.clip(z, -30, 30)
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical artifacts might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Apply soft cost penalty: divide by square root of cost
    # This is less aggressive than linear cost division, allowing exploration
    # of higher-cost regions if they offer higher expected improvement
    cost_safe = np.maximum(cost, 1e-8)
    scores = ei / np.sqrt(cost_safe)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
