"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with a small exploitation parameter xi.
    # xi > 0 makes the acquisition function more greedy.
    xi = 0.1
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the improvement
    improvement = f_best - mu - xi
    
    # Calculate z-score
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Calculate EI
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Return cost-aware score
    return ei / cost
# EVOLVE-BLOCK-END
