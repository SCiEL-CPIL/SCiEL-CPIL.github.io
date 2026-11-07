import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s kill

def _ensure_deadline():
    """Check if we're approaching the deadline."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return elapsed < (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement and
    Probability of Improvement with adaptive weighting.
    
    Uses a blend of EI and POI to balance exploration and exploitation.
    Also includes a deadline guard for safety.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    if not _ensure_deadline():
        # Fallback: return EI-based scores quickly
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (POI) for minimization
    # POI = P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Normalize EI and POI to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    
    # Adaptive weighting: favor POI when uncertainty is high, EI when exploitation is possible
    # Use a sigmoid-like function based on the relative magnitude of improvement
    # When f_best - mu is large relative to sigma, we're in exploitation regime -> favor EI
    # When f_best - mu is small relative to sigma, we're in exploration regime -> favor POI
    improvement_ratio = np.abs(f_best - mu) / sigma_safe
    # Weight for EI: increases with improvement ratio
    ei_weight = 1.0 / (1.0 + np.exp(-0.5 * (improvement_ratio - 2.0)))
    # Weight for POI: complement
    poi_weight = 1.0 - ei_weight
    
    # Blend EI and POI
    scores = ei_weight * ei_norm + poi_weight * poi
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
