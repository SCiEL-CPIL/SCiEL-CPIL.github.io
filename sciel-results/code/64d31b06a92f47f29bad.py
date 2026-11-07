# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach combining:
    1. Probability of improvement (POI) for exploration near the optimum
    2. Expected improvement (EI) for exploitation
    3. Cost-awareness via normalization
    
    The key insight is to use a threshold-based approach that balances
    exploration and exploitation while accounting for evaluation costs.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Hybrid score: combine POI and EI with a weighting that emphasizes
    # points with high probability of being better than current best
    # while also considering the magnitude of expected improvement
    
    # Use a geometric mean-like combination for robustness
    # This helps balance between finding any improvement (POI) 
    # and finding large improvements (EI)
    
    # Weight: favor points where POI is high (likely to be better)
    # but also reward large expected improvements
    weight = 0.5 + 0.5 * poi  # ranges from 0.5 to 1.0
    
    # Combined acquisition: weighted EI normalized by cost
    # The weight ensures we don't ignore regions with high POI but small EI
    score = weight * ei / cost
    
    # Add a small exploration bonus for high-uncertainty regions
    # that are far from current best (to avoid getting stuck)
    # This encourages exploring unexplored areas when uncertainty is high
    exploration_bonus = 0.1 * sigma_safe / (f_best - mu + 1e-9)
    exploration_bonus = np.clip(exploration_bonus, 0, 1.0) / cost
    
    # Final score
    final_score = score + exploration_bonus
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
