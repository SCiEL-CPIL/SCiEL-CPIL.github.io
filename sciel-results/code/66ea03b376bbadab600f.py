"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: balance exploration (sigma) and exploitation (f_best - mu)
    # with a penalty for high cost. Use a smooth approximation that avoids division by zero.
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost, but also add a small exploration bonus for high uncertainty
    # relative to the current best. This encourages exploring regions with high sigma.
    cost_safe = np.maximum(cost, 1e-8)
    
    # Normalize EI by cost
    score = ei / cost_safe
    
    # Add a small exploration term: regions with high sigma relative to f_best get a bonus
    # This helps escape local optima. The scale is set so it doesn't dominate EI.
    exploration_bonus = 0.05 * (sigma_safe / (np.abs(f_best) + 1e-6)) / cost_safe
    
    # Combine: primary is cost-aware EI, secondary is exploration bonus
    result = score + exploration_bonus
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
