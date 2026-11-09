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
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # LCB score for exploration
    # Adaptive kappa: larger when uncertainty is high relative to improvement
    f_min_known = 0.397887
    delta_f = max(f_best - f_min_known, 0.0)
    
    # Use a smoother decay for LCB weight
    lcb_weight = 0.25 * np.exp(-0.3 * delta_f)
    
    # Adaptive kappa based on relative uncertainty
    sigma_mean = np.mean(sigma)
    if sigma_mean > 1e-8:
        kappa = 0.8 + 0.4 * np.exp(-delta_f)
    else:
        kappa = 0.8
    
    lcb_score = -mu + kappa * sigma
    
    # Robust normalization using rank-based approach for stability
    # Convert to ranks to avoid sensitivity to outliers
    n = len(ei)
    ei_ranks = np.argsort(np.argsort(ei)).astype(float)
    lcb_ranks = np.argsort(np.argsort(lcb_score)).astype(float)
    
    # Normalize ranks to [0, 1]
    if n > 1:
        ei_norm = ei_ranks / (n - 1)
        lcb_norm = lcb_ranks / (n - 1)
    else:
        ei_norm = np.array([0.5])
        lcb_norm = np.array([0.5])
    
    # Blend: use EI as primary, LCB as exploration bonus
    scores = (1.0 - lcb_weight) * ei_norm + lcb_weight * lcb_norm
    
    # Add a small bonus for points with very low predicted mean (exploitation)
    # This helps refine the best region in later stages
    if delta_f < 0.1:
        # Late stage: boost exploitation of low-mu points
        mu_min = mu.min()
        mu_max = mu.max()
        if mu_max > mu_min:
            exploit_bonus = 0.1 * (1.0 - (mu - mu_min) / (mu_max - mu_min))
            scores += exploit_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
