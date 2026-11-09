"""Hybrid EI-LCB acquisition for Hartmann 3D Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends Expected Improvement (exploitation) with
    Lower Confidence Bound (exploration) for balanced search.
    
    Score = EI + kappa * (sigma - mu_normalized)
    where the LCB term encourages exploration in high-uncertainty regions
    while EI drives exploitation of promising areas.
    """
    t_start = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    n = mu.shape[0]
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    f_best_safe = f_best if np.isfinite(f_best) else 0.0
    
    # --- Expected Improvement (exploitation) ---
    z_ei = (f_best_safe - mu_safe) / sigma_safe
    ei = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # --- Lower Confidence Bound component (exploration) ---
    # For minimization: we want low mu - kappa*sigma
    # Score contribution: kappa * sigma (higher uncertainty = higher score)
    # But we also want to penalize very high mu, so we use: kappa * sigma - alpha * mu
    # Actually, the standard LCB score for minimization is: -mu + kappa*sigma
    # Let's use a moderate kappa that balances exploration
    kappa = 2.0
    lcb_score = kappa * sigma_safe
    
    # --- Blend: EI provides exploitation signal, LCB provides exploration ---
    # Normalize EI to have comparable scale to LCB term
    # EI is in objective units, LCB is in objective units (sigma is std dev)
    # They should be roughly comparable in scale for Hartmann 3D
    # Use a weighted sum with EI dominant but LCB providing exploration boost
    w_ei = 0.6
    w_lcb = 0.4
    score = w_ei * ei + w_lcb * lcb_score
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e6)
    
    # Time check (should be fast, but guard just in case)
    if time.time() - t_start > DEADLINE:
        # Fallback: return EI only
        ei = np.where(np.isfinite(ei), ei, -1e6)
        return ei
    
    return score
# EVOLVE-BLOCK-END
