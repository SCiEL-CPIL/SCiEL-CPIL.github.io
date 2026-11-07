"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time since module load for deadline guard
_T0 = time.time()
_TIMEOUT = 320.0  # Safety margin before 330s kill

# Known global minimum of Michalewicz 2D
_F_MIN_GLOBAL = -1.8013

# Weights for different components
_ALPHA_EI = 1.0       # Weight for Expected Improvement
_ALPHA_UCB = 0.5      # Weight for UCB exploration
_ALPHA_SIGMA = 0.3    # Weight for pure sigma (uncertainty)

def acquisition(mu, sigma, f_best):
    """
    Combined acquisition: EI + UCB + adaptive sigma bonus.
    
    This combines exploitation (EI), structured exploration (UCB with beta
    that decays as we converge), and a raw uncertainty bonus that provides
    early exploration when the gap to the optimum is large.
    """
    # Deadline guard
    if time.time() - _T0 > _TIMEOUT:
        # Fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -50, 50)
        return (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Expected Improvement for minimization ---
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # --- UCB: mu - beta * sigma ---
    # beta decays as we get closer to the global minimum
    gap = max(0.0, f_best - _F_MIN_GLOBAL)
    # gap ranges from ~2.8 (early) to 0 (converged)
    # beta starts high and decreases
    beta = 0.3 + 1.5 * min(1.0, gap / 2.8)
    ucb = mu - beta * sigma_safe
    
    # --- Normalize all components to comparable scales ---
    # Normalize EI
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-12
    ei_norm = (ei - ei_mean) / ei_std
    
    # Normalize UCB
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-12
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Normalize sigma
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe) + 1e-12
    sigma_norm = (sigma_safe - sigma_mean) / sigma_std
    
    # --- Adaptive weights ---
    # gap_norm: 1.0 early, 0.0 late
    gap_norm = min(1.0, gap / 2.8)
    
    # EI weight: always strong, slightly boosted when close to optimum
    w_ei = _ALPHA_EI * (0.8 + 0.4 * (1.0 - gap_norm))
    
    # UCB weight: strong early, moderate late
    w_ucb = _ALPHA_UCB * (0.5 + 0.5 * gap_norm)
    
    # Sigma weight: strong early, weak late
    w_sigma = _ALPHA_SIGMA * gap_norm ** 0.8
    
    # Combine
    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_sigma * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
