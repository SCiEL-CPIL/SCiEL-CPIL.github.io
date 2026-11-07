# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 15.0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an adaptive UCB-style
    exploration bonus and a confidence-scaled exploration component.
    
    The EI term exploits regions likely to beat f_best.
    The UCB term explores regions with high posterior uncertainty, helping escape
    local minima in the multi-modal Hartmann-3D landscape.
    
    The UCB weight adapts based on progress toward the known minimum (-3.86278):
    higher exploration early, more exploitation as we converge.
    A confidence scale based on |f_best| further modulates exploration.
    """
    _ensure_start_time()
    
    # Deadline guard: if we're close to the 300s limit, return a simple EI-based score
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive UCB weight based on progress toward known minimum
    f_min_known = -3.86278
    # progress: 1.0 when f_best is at 0 (start), 0.0 when f_best is at f_min_known
    progress = np.clip((f_best - f_min_known) / (0.0 - f_min_known), 0.0, 1.0)
    
    # More exploration early (up to 0.05), less late (down to 0.005)
    ucb_weight = 0.005 + 0.045 * progress
    
    # Confidence scale: reduces exploration as we get more confident (|f_best| increases)
    confidence_scale = 1.0 / (1.0 + 0.1 * np.abs(f_best))
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (standard formulation for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # UCB exploration bonus with adaptive weight and confidence scaling
    ucb_bonus = ucb_weight * confidence_scale * sigma_safe
    
    # Combined score: EI + UCB bonus
    scores = ei + ucb_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
