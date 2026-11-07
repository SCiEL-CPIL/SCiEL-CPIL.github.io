"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Budget-aware hybrid acquisition: interpolates between LCB (exploration)
    and EI (exploitation) based on remaining budget fraction.
    
    We don't know the exact step, but we can infer it from sigma magnitude
    or use a fixed heuristic. Since we have 30 total evaluations and 5 initial,
    we have 25 optimization steps. We'll use a simple heuristic:
    - Early on (high sigma relative to range), favor exploration (LCB)
    - Late on (low sigma), favor exploitation (EI)
    
    Actually, since we can't track steps directly, let's use the spread of sigma
    as a proxy: when sigma is large relative to the problem scale, we're in
    exploration mode; when small, exploitation mode.
    
    But a simpler approach: use a fixed blend that works well for Hartmann 6D.
    LCB with kappa ~ 2-3 worked well (score 1.178). Let's blend LCB and EI.
    
    Let's use: score = w * LCB + (1-w) * EI
    where w depends on how much uncertainty there is.
    
    For a robust approach: use LCB with adaptive kappa based on sigma distribution.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Lower Confidence Bound - use adaptive kappa
    # When sigma is large (uncertain), use higher kappa for exploration
    # When sigma is small (certain), use lower kappa for exploitation
    # Normalize sigma to [0, 1] range for adaptive kappa
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Adaptive kappa: higher when uncertainty is high
    # Base kappa around 2.5, modulated by uncertainty level
    kappa_base = 2.5
    # Use the mean normalized sigma as a global uncertainty indicator
    mean_sigma_norm = np.mean(sigma_norm)
    # kappa ranges from 1.5 (exploit) to 4.0 (explore)
    kappa = 1.5 + 2.5 * min(1.0, mean_sigma_norm * 2.0)
    
    lcb = mu - kappa * sigma_safe
    
    # Blend: use LCB primarily, with EI as a secondary signal
    # Normalize both to comparable scales
    lcb_max = np.max(lcb) - np.min(lcb)
    if lcb_max > 1e-9:
        lcb_norm = (lcb - np.min(lcb)) / lcb_max
    else:
        lcb_norm = np.zeros_like(lcb)
    
    ei_max = np.max(ei) - np.min(ei)
    if ei_max > 1e-9:
        ei_norm = (ei - np.min(ei)) / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Weight: favor LCB more in early stages (high uncertainty)
    w_lcb = 0.6 + 0.4 * min(1.0, mean_sigma_norm * 3.0)
    w_ei = 1.0 - w_lcb
    
    score = w_lcb * lcb_norm + w_ei * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
