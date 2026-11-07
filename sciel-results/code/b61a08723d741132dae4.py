# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: balance expected improvement against evaluation cost.
    # Use a robust variant of expected improvement per unit cost with an exploration
    # bonus to avoid over-concentration.
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Standard normal variables
    z = (f_best - mu) / sigma_safe
    
    # Expected improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI divided by cost, with a small exploration bonus
    # The exploration bonus encourages visiting high-uncertainty regions
    # when the expected improvement is low
    cost_safe = np.maximum(cost, 1e-6)
    
    # Base score: EI per unit cost
    score = ei / cost_safe
    
    # Add an exploration component: sigma per unit cost, weighted
    # This helps escape local optima in the acquisition landscape
    exploration_weight = 0.1
    score += exploration_weight * sigma_safe / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
