"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0

MIN_POSSIBLE = -0.8
MAX_POSSIBLE = 1.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with adaptive exploration.
    
    Structural Change:
    Replaces UCB with a refined Cost-Aware Expected Improvement (cEI) that:
    1. Properly computes EI for minimization
    2. Divides by cost for cost-efficiency
    3. Adds adaptive exploration based on distance to known minimum
    4. Uses numerical stability guards throughout
    
    This approach is theoretically grounded in regret minimization while
    explicitly accounting for evaluation costs.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1.0)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    # EI for minimization: improvement when f(x) < f_best
    # EI = (f_best - mu) * P(f < f_best) + sigma * E[(f_best - f) | f < f_best] * P(f < f_best)
    # Simplified: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = diff * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost
    c_ei = ei / cost_safe
    
    # Adaptive exploration: add a small bonus for high uncertainty regions
    # Scale exploration based on how far f_best is from the known minimum
    f_range = MAX_POSSIBLE - MIN_POSSIBLE
    if f_range <= 0:
        r = 0.5
    else:
        r = (f_best - MIN_POSSIBLE) / f_range
        r = np.clip(r, 0.0, 1.0)
    
    # Exploration weight: higher when f_best is poor (r close to 1)
    # This encourages exploration early, exploitation later
    exploration_weight = 0.1 * (1.0 - r)
    
    # Add exploration term: reward high sigma in cheap regions
    exploration_term = exploration_weight * sigma_safe / cost_safe
    
    # Final score: cost-aware EI + exploration
    score = c_ei + exploration_term
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
