"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 310.0
_CALL_COUNT = 0


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Phase-based acquisition for Hartmann 3D.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviations (2000,).
        f_best: Best observed value so far.
        
    Returns:
        Array of scores (2000,), where higher score means better candidate.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    # Check time budget
    if _time_remaining() < 5.0:
        # Emergency: return EI-based scores quickly
        z = (f_best - mu) / np.maximum(sigma, 1e-10)
        ei = sigma * norm.pdf(z) + (f_best - mu) * norm.cdf(z)
        return ei
    
    # Phase 1: Early exploration (calls 1-10) - Use EI with exploration bonus
    # Phase 2: Late exploitation (calls 11+) - Pure EI
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / np.maximum(sigma, 1e-10)
    ei = sigma * norm.pdf(z) + (f_best - mu) * norm.cdf(z)
    
    # Clip negative EI to zero
    ei = np.maximum(ei, 0.0)
    
    if _CALL_COUNT <= 10:
        # Early phase: Add exploration bonus proportional to uncertainty
        # The bonus decays as f_best improves (gets closer to optimal)
        # Hartmann 3D minimum is -3.86278, so progress = (f_best - (-3.86278)) / (initial_range)
        # We use a simpler approach: bonus scales with sigma but bounded relative to EI
        
        # Exploration bonus: proportional to sigma, but capped to avoid dominating EI
        # Use a coefficient that decays with call count
        decay = max(0.0, 1.0 - (_CALL_COUNT - 1) / 10.0)  # Linear decay from 1.0 to 0.0
        exploration_bonus = decay * 0.5 * sigma
        
        scores = ei + exploration_bonus
    else:
        # Late phase: Pure EI for fine-tuning
        scores = ei
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
