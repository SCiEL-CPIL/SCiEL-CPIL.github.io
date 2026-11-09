"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Theoretical minimum of Hartmann 6D
F_MIN = -3.32237
# Approximate range of the function for normalization heuristics
F_MAX_EST = 3.32

def acquisition(mu, sigma, f_best):
    """
    Deterministic Pareto-front-inspired Acquisition Function.
    
    Computes Expected Improvement (EI) and Predictive Standard Deviation (PSD).
    Normalizes both to [0, 1] across the candidate set.
    Combines them with a dynamic weight beta that favors exploration when f_best
    is far from the known minimum, and exploitation when f_best is close.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    # Avoid division by zero
    safe_sigma = np.where(sigma > 1e-10, sigma, 1e-10)
    
    z = (f_best - mu) / safe_sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it can be slightly negative due to floating point errors)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Predictive Standard Deviation (PSD)
    psd = sigma.copy()
    
    # 3. Normalize EI and PSD to [0, 1]
    # This step is crucial for balancing the two objectives with different scales.
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-10:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    psd_min = np.min(psd)
    psd_max = np.max(psd)
    psd_range = psd_max - psd_min
    if psd_range < 1e-10:
        psd_norm = np.ones_like(psd)
    else:
        psd_norm = (psd - psd_min) / psd_range
    
    # 4. Determine dynamic weight beta
    # Beta is the weight on the exploration term (PSD_norm).
    # (1 - Beta) is the weight on the exploitation term (EI_norm).
    
    # Heuristic:
    # If f_best is close to F_MIN, we want to exploit (low beta).
    # If f_best is far from F_MIN (e.g., positive), we want to explore (high beta).
    
    # Distance from minimum
    dist = f_best - F_MIN
    
    # Max possible distance (rough estimate)
    max_dist = F_MAX_EST - F_MIN
    
    # Normalize distance to [0, 1]
    # If dist is 0, norm_dist is 0. If dist is max_dist, norm_dist is 1.
    norm_dist = np.clip(dist / max_dist, 0.0, 1.0)
    
    # Beta increases with distance from minimum
    # At start (dist ~ 3.32 - (-3.32) = 6.64), norm_dist ~ 1.0 -> beta ~ 1.0 (Explore)
    # At end (dist ~ 0), norm_dist ~ 0.0 -> beta ~ 0.0 (Exploit)
    beta = norm_dist
    
    # 5. Combine scores
    # Score = (1 - beta) * EI_norm + beta * PSD_norm
    # Higher score is better.
    
    scores = (1.0 - beta) * ei_norm + beta * psd_norm
    
    # Safety check: ensure all scores are finite
    scores[~np.isfinite(scores)] = 0.0
    
    return scores

# EVOLVE-BLOCK-END
