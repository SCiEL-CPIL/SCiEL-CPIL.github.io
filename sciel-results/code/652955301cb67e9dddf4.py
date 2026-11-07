"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Use Expected Improvement (EI) per unit cost.
    # EI considers both the probability and the magnitude of improvement.
    # This is often more effective than POI for finding the global minimum.
    
    # Calculate z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    # EI = (f_best - mu) * P(f(x) < f_best) + sigma * pdf(z)
    # P(f(x) < f_best) = Phi(z)
    # pdf(z) = phi(z)
    improvement = f_best - mu
    prob_improvement = norm.cdf(z)
    density = norm.pdf(z)
    
    ei = improvement * prob_improvement + sigma * density
    
    # Ensure EI is non-negative (it should be, but numerical issues might arise)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: higher EI per unit cost is better
    score = ei / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
