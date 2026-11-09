"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return EI-based score
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Structural Change: Use a hybrid of Expected Improvement (EI) and 
    # Probability of Improvement (PI) with an exploration bonus based on
    # the posterior uncertainty relative to the current best.
    #
    # The key insight is that pure EI can be too exploitative in the early
    # stages when the GP posterior is still very uncertain. By blending
    # with PI and adding a scaled exploration term, we encourage the search
    # to explore regions with high uncertainty that may contain better
    # optima, while still favoring points with high expected improvement.
    #
    # We also use a "virtual" f_best that is slightly below the observed
    # f_best to encourage exploration away from the current best.

    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z_clipped)
    
    # Exploration bonus: scale sigma by a factor that encourages exploration
    # when the posterior is uncertain relative to the range of observed values.
    # We use a fixed exploration weight that balances exploitation and exploration.
    exploration_weight = 0.5
    
    # Hybrid score: blend EI and PI, with an exploration bonus
    # The exploration bonus is proportional to sigma, encouraging the search
    # to visit regions with high uncertainty.
    score = ei + exploration_weight * pi * sigma
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
