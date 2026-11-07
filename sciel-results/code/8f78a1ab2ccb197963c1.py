"""Model-and-optimize acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 20.0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a model-and-optimize approach with learned parameters.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Check deadline guard
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
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
    
    # Adaptive acquisition: switch between exploitation and exploration
    # based on how uncertain the posterior is relative to the current best.
    # When uncertainty is high relative to the gap, favor exploration (UCB-like).
    # When uncertainty is low, favor exploitation (EI).
    
    # Normalize sigma to [0, 1] for weighting
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-15:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # The "gap" is how far below f_best the predicted mean is
    gap = np.maximum(f_best - mu, 0.0)
    gap_norm = gap / (f_best - np.min(mu) + 1e-15) if f_best > np.min(mu) else np.zeros_like(mu)
    
    # Blend coefficient: high when uncertainty dominates, low when exploitation is clear
    # Use a smooth transition based on the ratio sigma / (gap + sigma)
    blend = sigma_safe / (sigma_safe + gap + 1e-10)
    
    # UCB-like score for exploration
    beta = 2.0 + 1.0 * (1.0 - np.mean(blend))  # adapt beta to overall uncertainty level
    ucb_score = mu + beta * sigma_safe
    
    # For minimization, we want to score high where we expect improvement
    # EI is already good for exploitation; for exploration, use negative UCB (lower is better)
    # But we need higher scores to mean "evaluate next", so:
    # - Exploitation: use EI directly (higher = better)
    # - Exploration: use -mu + beta*sigma (higher = more promising in uncertain regions)
    
    # Actually, let's use a cleaner formulation:
    # score = blend * exploration_score + (1 - blend) * ei
    # where exploration_score encourages high sigma and low mu
    
    # Exploration score: want low mu and high sigma
    # Normalize mu and sigma to comparable scales
    mu_range = np.max(mu) - np.min(mu) + 1e-15
    mu_norm = (mu - np.min(mu)) / mu_range  # 0 at best, 1 at worst
    
    # Exploration score: penalize high mu, reward high sigma
    exploration_score = (1.0 - mu_norm) + 0.5 * sigma_norm
    
    # Combine with smooth blending
    scores = blend * exploration_score + (1.0 - blend) * ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
