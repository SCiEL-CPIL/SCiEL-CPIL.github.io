"""Acquisition function for Bayesian optimization of sphere 1d."""
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

    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-8)

    # Pure Expected Improvement for minimization
    # This is a structural change from the hybrid EI+UCB approach to pure EI,
    # which is well-suited for 1D optimization where the GP posterior is informative.
    
    z = (f_best - mu) / s
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + s * pdf_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)

    return ei
# EVOLVE-BLOCK-END
