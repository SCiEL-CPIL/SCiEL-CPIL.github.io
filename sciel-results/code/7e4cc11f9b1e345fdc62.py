# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining:
    1. Expected Improvement (EI) - standard Bayesian optimization
    2. Uncertainty bonus - to encourage exploration
    3. Cost normalization - to prefer cheaper evaluations
    
    The acquisition value is higher for points that are:
    - Likely to improve upon f_best (high EI)
    - Have high uncertainty (exploration)
    - Are cheap to evaluate (low cost)
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance from f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (EI should be >= 0 theoretically, but numerical issues can cause negatives)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus: points with higher sigma are more uncertain
    # We use a square root to dampen the effect
    uncertainty_bonus = np.sqrt(sigma_safe)
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    # Add a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Combine EI and uncertainty bonus
    # The uncertainty bonus helps explore regions where the GP is uncertain
    combined = ei + 0.05 * uncertainty_bonus
    
    # Normalize by cost
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
