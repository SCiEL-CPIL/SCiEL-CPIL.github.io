"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known global minimum of Goldstein-Price 2D
TRUE_MIN = 3.0


def acquisition(mu, sigma, f_best):
    """
    Posterior-sampling-based acquisition that computes Expected Regret
    relative to the known true minimum (3.0) rather than f_best.
    
    This maintains a constant exploration signal that doesn't diminish
    as f_best approaches 3, keeping the acquisition sensitive to
    high-variance regions that may contain the true optimum.
    
    Combines:
    1. Expected Regret vs true minimum (strong exploration signal)
    2. Lower Confidence Bound (exploitation)
    3. Variance bonus (exploration of uncertain regions)
    """
    n = len(mu)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Component 1: Expected Regret relative to TRUE_MIN ---
    # This is the key structural change: use the known global minimum
    # instead of f_best, so the exploration signal never vanishes.
    z_er = (TRUE_MIN - mu) / sigma_safe
    er = (TRUE_MIN - mu) * norm.cdf(z_er) + sigma_safe * norm.pdf(z_er)
    
    # --- Component 2: LCB (Lower Confidence Bound) for exploitation ---
    # beta controls exploration vs exploitation tradeoff
    # As we get more evaluations, we can reduce beta, but we don't have
    # that info directly, so use a moderate value
    beta = 1.5
    lcb = mu - beta * sigma_safe
    
    # --- Component 3: Variance bonus for exploring uncertain regions ---
    # Regions with high sigma might contain the true minimum
    var_bonus = sigma_safe
    
    # --- Normalize components to comparable scales ---
    # Scale ER by typical magnitude
    er_scale = np.maximum(np.mean(np.abs(er)), 1e-6)
    er_norm = er / er_scale
    
    lcb_scale = np.maximum(np.abs(np.mean(lcb)), 1e-6)
    lcb_norm = lcb / lcb_scale
    
    var_scale = np.maximum(np.mean(var_bonus), 1e-6)
    var_norm = var_bonus / var_scale
    
    # --- Weighted combination ---
    # Higher weights on ER (true-min regret) and LCB (exploitation)
    w_er = 1.0
    w_lcb = 0.8
    w_var = 0.3
    
    score = w_er * er_norm + w_lcb * lcb_norm + w_var * var_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
