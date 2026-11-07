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
    
    # Global budget-aware scaling strategy
    # Instead of dividing by cost point-wise (which can amplify noise or distort scale),
    # we compute the cost z-score (how expensive each candidate is relative to the batch mean)
    # and subtract a penalty term. This encourages selecting candidates that are not just
    # high EI, but also relatively cheap compared to the other candidates in the batch.
    
    cost_mean = np.mean(cost)
    cost_std = np.std(cost)
    if cost_std < 1e-9:
        cost_std = 1.0
        
    cost_z = (cost - cost_mean) / cost_std
    
    # The penalty weight should be small enough not to dominate EI, but large enough
    # to break ties or slightly prefer cheaper options when EI is similar.
    # We scale the penalty by the typical magnitude of EI to make it dimensionless-ish.
    ei_scale = np.std(ei) if np.std(ei) > 1e-9 else 1.0
    
    # Penalty coefficient: 0.5 * EI_scale. This means a candidate 1 std dev more expensive
    # loses 0.5 * EI_scale in score. Since EI varies by EI_scale, this is a moderate penalty.
    penalty = 0.5 * ei_scale * cost_z
    
    scores = ei - penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
