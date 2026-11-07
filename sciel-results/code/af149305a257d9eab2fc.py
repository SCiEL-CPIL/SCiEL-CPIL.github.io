"""Rewritten acquisition function using Probability of Improvement with exploration bonus."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) with a small exploration bonus.
    
    POI: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    We add a small exploration bonus proportional to sigma to encourage
    visiting uncertain regions, which can help escape local optima in
    the Goldstein-Price function.
    """
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance from best
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    poi = norm.cdf(z)
    
    # Small exploration bonus: prefer higher uncertainty
    # Scale by 0.05 to keep it a minor perturbation
    exploration = 0.05 * sigma_safe
    
    # Combine: POI dominates, exploration breaks ties and encourages exploration
    score = poi + exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
