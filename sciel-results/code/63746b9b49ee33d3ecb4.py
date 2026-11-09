"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: combines expected improvement per unit cost
    with an uncertainty bonus to navigate flat Griewank landscape.
    
    Strategy:
    - Base score: EI / cost (exploit high-improvement, low-cost regions)
    - Bonus: sigma / cost (explore high-uncertainty regions where minima may be hidden)
    - Smooth blend to avoid extreme values
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Uncertainty bonus: reward regions with high predictive variance
    # This helps navigate flat landscapes where EI is near zero
    # but there might be undiscovered minima
    sigma_cost = sigma_safe / cost_safe
    
    # Combine: primary signal is EI/cost, secondary is uncertainty/cost
    # Use a weighted sum with the uncertainty term scaled down
    # The scale of 0.5 balances exploration vs exploitation
    score = ei_cost + 0.5 * sigma_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
