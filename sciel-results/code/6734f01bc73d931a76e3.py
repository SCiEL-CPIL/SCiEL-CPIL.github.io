"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    - Early on: more exploration via LCB with higher beta
    - Later: more exploitation via EI
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound for minimization: -mu - beta * sigma
    # Use adaptive beta that decays over time
    # We don't have explicit step count, so use a heuristic based on how far mu is from f_best
    # When we're close to optimum (small sigma), we want more exploitation
    # When uncertainty is high, we want more exploration
    
    # Adaptive beta: higher when sigma is large relative to the spread of f_best - mu
    # Scale beta based on the typical uncertainty
    typical_sigma = np.median(sigma_safe)
    if typical_sigma < 1e-6:
        beta = 0.1
    else:
        # Start with higher exploration, decay as we get more confident
        # Use a simple decay based on how many points have low uncertainty
        low_unc_frac = np.mean(sigma_safe < 0.1 * typical_sigma)
        beta = 1.5 * (1.0 - 0.5 * low_unc_frac)
    
    lcb = -(mu + beta * sigma_safe)
    
    # Normalize both to comparable scale
    # EI is non-negative, LCB is negative
    # Blend them: use EI primarily but add exploration bonus
    # Actually, let's use a weighted combination
    
    # Scale EI to be comparable to LCB
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_normalized = ei / ei_max
    
    # For LCB, higher (less negative) is better
    lcb_max = np.max(lcb)
    lcb_min = np.min(lcb)
    if lcb_max - lcb_min > 1e-9:
        lcb_normalized = (lcb - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_normalized = np.zeros_like(lcb)
    
    # Weight: start with more exploration (LCB), shift to exploitation (EI)
    # Use the fraction of low-uncertainty points as a proxy for progress
    progress = np.mean(sigma_safe < 0.2 * typical_sigma)
    w_ei = 0.5 + 0.5 * progress  # 0.5 to 1.0
    w_lcb = 1.0 - w_ei + 0.3  # Keep some exploration always
    
    scores = w_ei * ei_normalized + w_lcb * lcb_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
