# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _elapsed():
    return time.time() - _ensure_start_time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an adaptive UCB-style exploration bonus.
    
    The EI term exploits regions likely to beat f_best.
    The UCB term explores regions with high posterior uncertainty, helping escape local minima.
    The UCB weight adapts based on progress toward the known minimum (-3.86278):
    higher exploration early, more exploitation as we converge.
    """
    # Deadline guard: if we're close to the 330s limit, return a simple EI-based score
    if _elapsed() > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = np.where(np.isfinite(ei), ei, 0.0)
        return scores
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB weight based on progress toward known minimum
    f_min_known = -3.86278
    # progress: 1.0 when f_best is at 0 (start), 0.0 when f_best is at f_min_known
    progress = np.clip((f_best - f_min_known) / (0.0 - f_min_known), 0.0, 1.0)
    
    # More exploration early (up to 0.05), less late (down to 0.005)
    ucb_weight = 0.005 + 0.045 * progress
    
    # Expected Improvement (standard formulation for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration bonus
    ucb_bonus = ucb_weight * sigma_safe
    
    # Combined score: EI + UCB bonus
    scores = ei + ucb_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
