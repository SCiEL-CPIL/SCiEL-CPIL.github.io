# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of:
    1. Expected Improvement (EI) per unit cost
    2. Probability of Improvement (PI) bonus
    3. Uncertainty bonus (to encourage exploration)
    
    This creates a more robust exploration-exploitation tradeoff than pure EI/cost.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (how likely is x to be better than f_best?)
    pi = norm.cdf(z)
    
    # Uncertainty term: reward high uncertainty regions
    # Use a scaled version of sigma to balance exploration
    uncertainty_bonus = sigma_safe * 0.1
    
    # Combine: primary signal is EI/cost, but add PI and uncertainty terms
    # The PI term helps when EI is small but probability of improvement is decent
    # The uncertainty term encourages exploring unexplored regions
    
    # Normalize cost to [0, 1] range for better scaling
    cost_safe = np.maximum(cost, 1e-9)
    
    # Main acquisition: EI per cost
    ei_per_cost = ei / cost_safe
    
    # Add a weighted probability of improvement term (also cost-normalized)
    pi_per_cost = pi / cost_safe * 0.5
    
    # Add uncertainty bonus (not cost-normalized, to encourage exploration even at higher cost)
    total = ei_per_cost + pi_per_cost + uncertainty_bonus
    
    # Ensure finite values
    total = np.where(np.isfinite(total), total, 0.0)
    
    return total
# EVOLVE-BLOCK-END
