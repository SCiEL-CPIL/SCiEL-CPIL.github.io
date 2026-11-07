"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with a quadratic cost penalty.
    
    The score is calculated as EI(x) / cost(x)^2. This formulation strongly favors
    evaluations in low-cost regions while still seeking high expected improvement.
    The quadratic penalty on cost creates a sharper exploration-exploitation balance,
    directing the search more aggressively towards the cheap central regions where
    the griewank minimum is likely to be found (given the shift constraints).
    """
    t0 = time.time()
    
    # Ensure inputs are valid and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm for numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative by definition, but numerical errors can cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: Quadratic penalty on cost
    # This amplifies the preference for low-cost points compared to linear normalization
    score = ei / (cost_safe ** 2)
    
    # Clamp non-finite values to 0 to ensure validity
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - t0 > 280:
        return score
        
    return score
# EVOLVE-BLOCK-END
