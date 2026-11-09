"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 330  # 330s limit, keep margin

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE - 10:
        # Fallback: return zero or simple heuristic to ensure return
        # If we are close to deadline, just return EI / cost to be safe and fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # 1. Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Handle numerical stability for very large z
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # 2. Adaptive Cost Penalty
    # Instead of dividing by cost, we subtract a scaled cost penalty.
    # This avoids singularities and allows high-EI points with higher cost to be selected
    # if the EI gain justifies the cost.
    
    # Scale the cost penalty to be comparable to the EI spread
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range < 1e-9:
        # If all costs are the same, cost penalty is irrelevant, just return EI
        return ei
        
    # We want the cost penalty to vary over a similar range as the EI
    # Let's set the penalty such that the difference in cost between min and max cost
    # corresponds to a fraction of the EI range.
    # A factor of 0.5 means the cost penalty spans half the EI range.
    # This is a heuristic balance.
    lambda_cost = 0.5 * ei_range / cost_range
    
    # Apply penalty
    # We subtract the penalty. Since we want to maximize score,
    # higher cost should lower the score.
    score = ei - lambda_cost * cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
