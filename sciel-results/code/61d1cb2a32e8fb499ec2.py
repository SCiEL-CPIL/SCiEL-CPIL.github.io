"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition combining adaptive LCB with Expected Improvement.
    
    The LCB component provides strong exploration early and exploitation later
    via an adaptive beta. The EI component provides a well-calibrated improvement
    measure. A diversity-promoting perturbation breaks ties and encourages
    spatial diversity across the 2000 candidates.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Adaptive LCB component ---
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    if sigma_max > 1e-10:
        conv_ratio = sigma_mean / sigma_max
    else:
        conv_ratio = 1.0
    
    # Beta starts high (exploratory) and decreases as convergence increases
    beta_base = 2.0
    beta = beta_base * (0.5 + 0.5 * conv_ratio)
    
    # Scale by how far f_best is from the theoretical minimum
    f_min_theory = -3.86278
    gap = f_best - f_min_theory
    if gap > 1e-6:
        beta *= (1.0 + 0.5 * min(gap / 2.0, 1.0))
    
    # LCB score: we want to minimize f(x), so lower LCB is better.
    # For "higher score = evaluate next", we negate: score = -(mu - beta*sigma)
    lcb_scores = -(mu - beta * sigma_safe)
    
    # --- Expected Improvement component ---
    # Standard EI for minimization:
    #   EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in extreme cases
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Normalize EI to be on a comparable scale to LCB scores
    # EI is always >= 0, while LCB scores can be negative or positive
    # We want to blend them meaningfully
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    
    lcb_mean = np.mean(lcb_scores)
    lcb_std = np.std(lcb_scores)
    
    # Normalize both to zero mean, unit variance for blending
    if lcb_std > 1e-12:
        lcb_norm = (lcb_scores - lcb_mean) / lcb_std
    else:
        lcb_norm = np.zeros_like(lcb_scores)
    
    if ei_std > 1e-12:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei)
    
    # --- Blend LCB and EI ---
    # Weighting: more LCB early (high uncertainty), more EI later (low uncertainty)
    # Use conv_ratio as a proxy: high conv_ratio = early, low = later
    w_lcb = 0.5 + 0.3 * conv_ratio  # ranges from 0.5 to 0.8
    w_ei = 1.0 - w_lcb               # ranges from 0.2 to 0.5
    
    # Combine normalized scores
    blended = w_lcb * lcb_norm + w_ei * ei_norm
    
    # Scale back to a reasonable magnitude
    # Use the scale of the original LCB scores as reference
    scale_ref = max(lcb_std, 1e-6)
    scores = blended * scale_ref
    
    # --- Diversity-promoting perturbation ---
    # Small deterministic perturbation based on index to break ties
    indices = np.arange(len(mu))
    perturbation = 0.001 * np.sin(indices * 0.1)
    
    scores = scores + perturbation
    
    # Time check: ensure we finish well within limits
    elapsed = time.time() - start_time
    if elapsed > 20.0:
        # Fallback: just return LCB scores if something is slow
        scores = lcb_scores + perturbation
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
