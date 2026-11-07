"""Direct acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're getting close to the limit, return a safe fallback
    elapsed = time.time() - _T0
    if elapsed > _DEADLINE:
        # Return scores that favor low mu (exploitation) to be safe
        return -mu.copy()
    
    # Ensure sigma is positive and finite
    s = np.maximum(sigma, 1e-8)
    s = np.where(np.isfinite(s), s, 1e-8)
    
    # Use Expected Improvement (EI) with a small exploration parameter xi
    # For sphere function minimization, EI is standard
    # Using xi=0.01 to balance exploitation and exploration
    
    xi = 0.01
    z = (f_best - mu - xi) / s
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu - xi) * cdf_z + s * pdf_z
    
    # Ensure finite and non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
