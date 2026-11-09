"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used most of the 330s budget, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a quick fallback: PoI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        return poi / cost
    
    # Compute z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI component: expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # PoI component: probability of improvement
    poi = norm.cdf(z)
    
    # Adaptive blend:
    # - When sigma is large relative to (f_best - mu), we're in uncertain regions
    #   -> favor PoI for exploration
    # - When sigma is small and mu < f_best, we're near a good point
    #   -> favor EI for exploitation
    #
    # Use a soft weight based on the ratio sigma / max(|f_best - mu|, sigma)
    # This gives weight in [0, 1] where 1 = fully exploratory (PoI), 0 = fully exploitative (EI)
    
    # Scale: use the ratio of sigma to the absolute improvement potential
    improvement_potential = np.maximum(np.abs(f_best - mu), 1e-9)
    uncertainty_ratio = sigma_safe / (improvement_potential + sigma_safe)
    
    # uncertainty_ratio is in (0, 1): high when sigma >> |f_best - mu|
    # We want: high uncertainty_ratio -> more PoI, low -> more EI
    # But we also want a baseline level of exploration to avoid local optima
    
    # Blend weight: w = 0.3 + 0.7 * uncertainty_ratio
    # This ensures at least 30% PoI (exploration) and up to 100% PoI in highly uncertain regions
    w = 0.3 + 0.7 * uncertainty_ratio
    
    # Combine: w * PoI-based score + (1-w) * EI-based score
    # To keep units consistent, scale PoI by a typical improvement magnitude
    # Use: PoI_scaled = PoI * max(0, f_best - mu) when mu < f_best, else PoI * sigma
    # This makes PoI comparable in magnitude to EI
    
    # For regions where mu < f_best, the potential improvement is (f_best - mu)
    # For regions where mu >= f_best, use sigma as a proxy for potential gain
    potential_gain = np.where(mu < f_best, f_best - mu, sigma_safe)
    potential_gain = np.maximum(potential_gain, 1e-9)
    
    poi_scaled = poi * potential_gain
    
    # Blend
    blended = w * poi_scaled + (1 - w) * ei
    
    # Normalize by cost
    score = blended / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
