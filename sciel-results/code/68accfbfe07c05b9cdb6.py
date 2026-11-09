"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for time management
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we are too close to the limit
    elapsed = time.time() - _start_time
    if elapsed > 295:
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is strictly positive
    cost = np.maximum(cost, 1e-9)
    
    # z-score for Expected Improvement / Probability of Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Structural change: Use a non-linear cost-aware blending of EI and PI.
    # Instead of simple division by cost, use a cost penalty that grows sub-linearly.
    # This allows moderate-cost points with high EI to compete better.
    # 
    # The idea: Score = (w1 * EI + w2 * PI) / (1 + cost_penalty * log(1 + cost))
    # where log(1+cost) grows slower than cost, so expensive points are penalized
    # less aggressively than with linear 1/cost.
    #
    # Additionally, we add a small exploration bonus proportional to sigma to
    # encourage visiting uncertain regions even when EI is low.
    
    # Cost penalty: sub-linear growth
    cost_penalty = 0.5
    cost_factor = 1.0 + cost_penalty * np.log1p(cost)
    
    # Base score: blend of EI and PI
    # EI captures magnitude of improvement, PI captures probability
    # Weight EI more heavily since it's more informative about actual improvement
    base_score = 0.7 * ei + 0.3 * pi
    
    # Exploration bonus: reward uncertainty
    # Normalize sigma to a reasonable range
    sigma_bonus = 0.1 * sigma / (sigma + 0.1)
    
    # Combine
    scores = (base_score + sigma_bonus) / cost_factor
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
