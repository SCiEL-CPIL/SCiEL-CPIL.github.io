"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're getting close to the limit, return a safe fallback
    elapsed = time.time() - _T0
    if elapsed > _DEADLINE:
        # Return scores that favor low mu (exploitation) to be safe
        return -mu.copy()
    
    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) with xi=0 for pure exploitation
    # But we can also consider a small xi to encourage exploration
    # For sphere function minimization, EI is standard
    
    z = (f_best - mu) / s
    # Use scipy.stats.norm for accuracy
    from scipy.stats import norm
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + s * pdf_z
    
    # Ensure finite and non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
