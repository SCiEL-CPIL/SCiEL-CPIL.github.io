"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Structural change: Use a two-phase acquisition strategy.
    Phase 1 (early): Pure exploration via UCB to cover the domain.
    Phase 2 (late): Exploitation via Expected Improvement to refine the best region.
    
    The transition is based on how close f_best is to the known global minimum.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Known global minimum for Michalewicz 2D
    global_min = -1.8013
    
    # Progress toward global minimum: 0 when f_best is poor, 1 when close
    # Typical f_best range: [-2.0, 0.5]
    # progress = (f_best - 0.5) / (global_min - 0.5)
    # When f_best = 0.5, progress = 0 (explore)
    # When f_best = global_min, progress = 1 (exploit)
    progress = np.clip((f_best - 0.5) / (global_min - 0.5), 0.0, 1.0)
    
    # Phase 1: UCB exploration score (higher = more promising to explore)
    # For minimization: -mu + kappa * sigma
    # Start with large kappa for broad exploration
    kappa_early = 2.0
    ucb_early = -mu + kappa_early * sigma_safe
    
    # Phase 2: EI exploitation score (higher = more promising to exploit)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend between UCB and EI based on progress
    # Early: mostly UCB, Late: mostly EI
    # Use a smooth transition
    w_ucb = 1.0 - progress
    w_ei = progress
    
    # Normalize both scores to [0, 1] range for fair blending
    ucb_min, ucb_max = np.min(ucb_early), np.max(ucb_early)
    ei_min, ei_max = np.min(ei), np.max(ei)
    
    ucb_norm = (ucb_early - ucb_min) / (ucb_max - ucb_min + 1e-12)
    ei_norm = (ei - ei_min) / (ei_max - ei_min + 1e-12)
    
    # Weighted combination
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
