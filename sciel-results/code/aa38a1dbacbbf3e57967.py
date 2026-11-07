"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Cost-Aware Expected Improvement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Precompute constants
_EPS = 1e-9
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization:
       EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
       where z = (f_best - mu(x)) / sigma(x)
       This measures the expected reduction in the objective value.
       
    2. Normalize by the evaluation cost to prioritize cheap improvements.
       This is a standard approach in cost-aware Bayesian optimization.
       
    3. Return the cost-normalized EI scores (higher = better to evaluate next).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Deadline guard
    t0 = time.time()
    
    # 1. Compute Expected Improvement
    # For minimization, improvement is f_best - f(x)
    # E[max(0, f_best - f(x))] where f(x) ~ N(mu, sigma^2)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, _EPS)
    
    # z-score for the improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Use vectorized operations for speed
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative, but numerical errors might cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_MARGIN:
        # Fallback to simple cost-normalized EI
        cost_safe = np.maximum(cost, _EPS)
        return ei / cost_safe

    # 2. Cost-awareness
    # Divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, _EPS)
    
    final_score = ei / cost_safe
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Final deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return final_score
        
    return final_score

# EVOLVE-BLOCK-END
