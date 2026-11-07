"""Rewritten acquisition for Bayesian optimization of Hartmann-3D using model-and-optimize approach."""
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
    Acquisition function combining Expected Improvement with a UCB-style exploration bonus
    and a targeted exploration component for the multi-modal Hartmann-3D landscape.
    
    The EI term exploits regions likely to beat f_best.
    The UCB term (scaled by confidence) explores regions with high posterior uncertainty,
    helping escape local minima.
    
    We use a reduced UCB weight (0.03) with confidence scaling to allow more exploitation
    as the search converges, while still maintaining enough exploration to find global minima.
    """
    _ensure_start_time()
    
    # Check deadline guard
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        # Return a simple EI-based score quickly to stay within deadline
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Weight parameters optimized for Hartmann-3D scale
    # Slightly reduced UCB weight to favor exploitation once good candidates are found
    ucb_weight = 0.03
    log_ei_weight = 0.1
    
    # UCB component: scale down as we get more confident (based on f_best magnitude)
    # Hartmann-3D minimum is around -3.86, so f_best will be negative
    # Scale UCB by a factor that decreases as |f_best| increases (more confident)
    confidence_scale = 1.0 / (1.0 + 0.1 * np.abs(f_best))
    ucb_component = ucb_weight * confidence_scale * sigma_safe
    
    # Log-scaled EI to balance exploration/exploitation
    # Use a soft log to avoid issues with negative or zero values
    ei_shifted = np.maximum(ei, 1e-8)
    log_ei_component = log_ei_weight * np.log1p(ei_shifted)
    
    # Additional exploration bonus for points with high uncertainty and moderate EI
    # This helps escape local optima in the multi-modal landscape
    # Bonus peaks when EI is small but sigma is large (uncertain but not clearly bad)
    exploration_bonus = 0.02 * sigma_safe * np.exp(-0.5 * np.square((ei - 0.01) / 0.01))
    
    # Combine: primarily EI, with UCB, log-EI, and exploration bonus
    scores = ei + ucb_component + log_ei_component + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
