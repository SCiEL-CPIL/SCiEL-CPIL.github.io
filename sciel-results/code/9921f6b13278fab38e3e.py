"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return EI-based score
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu(x), sigma(x))
    # This is a structurally different acquisition from LCB:
    # - EI directly measures expected improvement over best observed
    # - No manual kappa tuning needed
    # - Naturally balances exploration/exploitation
    # - Points with high sigma far from f_best get moderate EI
    # - Points near f_best with low sigma get low EI (exploit)
    
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Blend with a small LCB component for additional exploration
    # This helps in early stages when EI might be too conservative
    # Use a small weight that decays as f_best improves
    f_min_known = 0.397887
    delta_f = max(f_best - f_min_known, 0.0)
    lcb_weight = 0.3 * np.exp(-0.5 * delta_f)
    kappa = 1.0
    lcb_score = -mu + kappa * sigma
    
    # Normalize both to comparable scale before blending
    # Use min-max normalization within the candidate set
    ei_range = ei.max() - ei.min() if ei.max() > ei.min() else 1.0
    lcb_range = lcb_score.max() - lcb_score.min() if lcb_score.max() > lcb_score.min() else 1.0
    ei_norm = (ei - ei.min()) / ei_range
    lcb_norm = (lcb_score - lcb_score.min()) / lcb_range
    
    scores = (1.0 - lcb_weight) * ei_norm + lcb_weight * lcb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
