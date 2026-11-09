"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Probability of Improvement (PoI) per unit cost.
    
    PoI is more robust than EI in high-variance regions (like the steep walls
    of the Rosenbrock valley) because it depends on the probability of beating
    f_best rather than the expected magnitude of improvement. This encourages
    steady navigation along the narrow valley floor rather than jumping to
    high-uncertainty regions.
    
    We normalize by cost to account for the cost-aware evaluation budget.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_clipped = np.maximum(sigma, 1e-9)
    
    # z-score for probability of improvement
    z = (f_best - mu) / sigma_clipped
    
    # PoI = P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Cost-aware: divide by cost to prioritize cheaper evaluations
    # cost is always >= 1 (1 + mean(x scaled to [0,1])), so this is safe
    scores = poi / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
