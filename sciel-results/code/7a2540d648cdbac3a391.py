# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300  # Seconds

def _check_time():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - 5)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a heavily exploitation-weighted blend of
    Expected Improvement (EI) and a normalized exploitation score.
    
    Based on insights that heavily weighting exploitation (-mu) improves
    performance on multi-modal landscapes like Hartmann 3D for small budgets.
    
    For minimization:
    - EI is computed as standard.
    - Exploitation score is (mu_max - mu) / (mu_max - mu_min), favoring low mu.
    - The final score is a weighted sum: 30% EI, 70% Exploitation.
    """
    # Safety check: if we are running out of time, return a safe fallback
    if _check_time():
        # Fallback: return -mu (pure exploitation) to ensure we finish
        return -mu

    # Handle near-zero sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for blending
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Exploitation score: prefer lower predicted values
    # Scale to be comparable with EI
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max > mu_min:
        exploit_score = (mu_max - mu) / (mu_max - mu_min)
    else:
        exploit_score = np.zeros_like(mu)
    
    # Blend: 30% EI, 70% exploitation
    # This creates a strong bias towards exploiting good regions,
    # guided by the insight that pure EI can be misleading in unexplored areas.
    alpha = 0.3
    score = alpha * ei_norm + (1 - alpha) * exploit_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
