"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) based acquisition function.
    
    Uses POI scaled by the magnitude of expected gain to balance exploration
    and exploitation. Prioritizes regions with high uncertainty that have a
    significant chance of beating the current best.
    
    The score is: POI * (f_best - mu) + sigma * pdf(z) * (f_best - mu)
    This emphasizes both the probability of improvement and the expected gain.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    poi = norm.cdf(z)
    
    # Expected improvement (EI) for reference
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # POI scaled by expected gain magnitude
    # This naturally balances: high POI points get boosted if they also have
    # large expected gains, while uncertain points with moderate POI still
    # contribute via the pdf term
    gain_scale = np.maximum(f_best - mu, 0)
    poi_scaled = poi * gain_scale
    
    # Add uncertainty bonus to encourage exploration in high-variance regions
    uncertainty_bonus = sigma_safe * norm.pdf(z)
    
    # Combine: emphasize POI with gain scaling, plus uncertainty exploration
    score = poi_scaled + 0.5 * uncertainty_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
