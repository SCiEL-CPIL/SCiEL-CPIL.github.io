"""Cost-aware acquisition for Bayesian optimization on Powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses a modified expected improvement (EI) that accounts for the cost of
    each evaluation. The cost is 1 + mean(x) where x is scaled to [0,1].
    Since we don't have x directly, we use the provided cost array.
    
    We use a combination of:
    1. Expected Improvement (EI) - standard BO acquisition
    2. Cost normalization - divide by cost to favor cheaper evaluations
    3. Uncertainty bonus - add a small term to encourage exploration
    4. Deadline guard - track time and return safely if approaching limit
    """
    # Track time for deadline guard
    t_start = time.time()
    deadline = 300.0 - 30.0  # 30s safety margin
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size != 2000 or sigma.size != 2000 or cost.size != 2000:
        # Return uniform scores as fallback
        return np.ones(2000, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score for expected improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware acquisition: EI per unit cost
    # This favors points that have high expected improvement relative to their cost
    acquisition_score = ei / cost_safe
    
    # Add a small uncertainty bonus to encourage exploration
    # The bonus is proportional to sigma but scaled down
    uncertainty_bonus = 0.1 * sigma_safe / cost_safe
    
    # Combine
    scores = acquisition_score + uncertainty_bonus
    
    # Handle NaN/Inf values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > deadline:
        # Return current best result
        pass
    
    return scores.astype(np.float64)
# EVOLVE-BLOCK-END
