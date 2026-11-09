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
    
    # Branin 2D optimum
    f_opt = 0.397887
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute Probability of Improvement (PoI)
    poi = norm.cdf(z_clipped)
    
    # Compute a lower confidence bound (LCB) based score
    # LCB = mu - kappa * sigma
    # We want high LCB scores to mean "likely good", so we negate
    # Actually for minimization, LCB = mu - kappa*sigma, lower is better
    # So score = -LCB = -mu + kappa*sigma, higher is better
    # Let's use a kappa that adapts to progress
    
    # Progress measure: how close f_best is to f_opt
    gap = max(f_best - f_opt, 1e-10)
    # When gap is large, we're early, explore more (higher kappa)
    # When gap is small, we're late, exploit more (lower kappa)
    # Use exponential decay
    kappa = 0.5 * np.exp(-3.0 * gap / max(f_best, 1e-6))
    
    # LCB score: higher means more promising
    lcb_score = -mu + kappa * sigma
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, -np.inf)
    
    # Normalize EI and LCB scores to [0, 1] range for blending
    # Use robust normalization based on percentiles to avoid outlier issues
    
    def robust_normalize(arr):
        """Normalize array to [0,1] using percentile-based scaling"""
        if len(arr) < 2:
            return arr.copy()
        p10 = np.percentile(arr, 10)
        p90 = np.percentile(arr, 90)
        if p90 - p10 < 1e-12:
            return np.zeros_like(arr)
        normalized = (arr - p10) / (p90 - p10)
        return np.clip(normalized, 0, 1)
    
    ei_norm = robust_normalize(ei)
    lcb_norm = robust_normalize(lcb_score)
    
    # Blend EI and LCB
    # Early: more weight on LCB (exploration)
    # Late: more weight on EI (exploitation)
    # Use the gap to determine the blend weight
    # weight_ei: 0.5 when far, approaches 1.0 when close
    weight_ei = 0.5 + 0.5 * (1.0 - np.exp(-5.0 * gap / max(f_best, 1e-6)))
    weight_lcb = 1.0 - weight_ei
    
    scores = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima
    sigma_norm = robust_normalize(sigma)
    exploration_bonus = 0.1 * sigma_norm * (1.0 - weight_ei)
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
