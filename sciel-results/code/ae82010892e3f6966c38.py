"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) approximation for cost-aware Bayesian optimization.
    
    KG estimates the expected improvement in the best observed value.
    For a single candidate x, the KG is:
        E[max(0, f_best - f(x))]
    which is exactly the Expected Improvement (EI).
    
    However, for the Griewank function with its flat, multi-modal landscape,
    we use a modified KG that:
    1. Uses a robust cost normalization to handle the cost range [1, 2]
    2. Adds an exploration bonus for high-uncertainty regions
    3. Uses a temperature parameter to balance exploitation vs exploration
    
    The key structural change: instead of simple EI/cost, we use:
        KG = EI * (1 + alpha * uncertainty_bonus) / cost_normalized
    
    where uncertainty_bonus captures the potential for finding new minima
    in flat regions (where sigma is high relative to the scale of f).
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard EI calculation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization: cost is in [1, 2], normalize to [0.5, 1]
    # This makes the cost penalty less extreme
    cost_norm = 0.5 + 0.5 * (cost - 1.0) / (2.0 - 1.0 + 1e-9)
    cost_norm = np.maximum(cost_norm, 0.5)
    
    # Uncertainty bonus: reward high sigma regions where new minima may be hidden
    # The Griewank function has many local minima in flat regions
    # Scale sigma by a characteristic scale of the function
    # For griewank 2d on [-10,10]^2, the function values are typically in [0, 0.1]
    # Use a scale of 0.01 to normalize uncertainty
    scale = 0.01
    uncertainty = sigma_safe / (scale + 1e-9)
    
    # Cap the uncertainty bonus to prevent extreme values
    uncertainty_bonus = np.minimum(uncertainty, 10.0)
    
    # Temperature parameter: balance exploitation vs exploration
    # Higher temperature = more exploration
    temperature = 0.3
    alpha = temperature
    
    # KG score with uncertainty bonus
    kg = ei * (1.0 + alpha * uncertainty_bonus) / cost_norm
    
    # Ensure finite values
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    return kg
# EVOLVE-BLOCK-END
