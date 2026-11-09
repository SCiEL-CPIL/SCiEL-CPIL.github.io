import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: normalized sigma
    # This encourages exploring regions with high uncertainty, scaled down
    # to not overwhelm the exploitation signal (EI).
    max_sigma = np.max(sigma_safe)
    if max_sigma > 0:
        exploration_bonus = 0.1 * (sigma_safe / max_sigma)
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
        
    # Combined score
    # We add the exploration bonus to EI.
    # EI is in objective units. The bonus is scaled to be comparable.
    # Since Hartmann 3D range is approx 4 units, 0.1 is a reasonable small nudge.
    score = (ei + exploration_bonus) / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
