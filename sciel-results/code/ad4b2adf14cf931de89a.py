"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Check time limit to ensure we return within constraints
    if time.time() - _start_time > 320:
        # Fallback: pure exploitation
        return -mu

    # Structural change: Use Upper Confidence Bound (UCB) with adaptive exploration
    # UCB(x) = mu(x) + kappa * sigma(x) for minimization, but we want to MINIMIZE,
    # so we use: score = -mu(x) + kappa * sigma(x)
    # This balances exploitation (low mu) with exploration (high sigma)
    
    # Adaptive kappa based on posterior uncertainty
    # When sigma is high overall, explore more; when low, exploit more
    mean_sigma = np.mean(sigma)
    
    # Scale kappa: start with moderate exploration, reduce as uncertainty decreases
    # Use a sigmoid-like scaling to keep kappa in a reasonable range
    # When mean_sigma is large (early on), kappa ~ 1.0; when small (late), kappa ~ 0.1
    base_kappa = 1.0
    min_kappa = 0.1
    # Normalize mean_sigma to get a scaling factor
    # Assume typical sigma range for this problem is around 0.01 to 1.0
    norm_sigma = np.clip(mean_sigma / 0.5, 0, 1)  # Normalize to [0, 1]
    kappa = min_kappa + (base_kappa - min_kappa) * norm_sigma
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute UCB score: higher is better for acquisition
    # For minimization, we want low mu and high sigma (uncertainty)
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
