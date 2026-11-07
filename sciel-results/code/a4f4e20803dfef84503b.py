import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pure Expected Improvement (EI) for minimization.
    
    Strategy:
    1. Calculate EI for each candidate point.
    2. EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    3. Return the EI scores. Higher scores indicate better candidates.
    
    This approach is preferred over blended UCB/EI because:
    - It directly optimizes the expected reduction in the objective function.
    - It naturally balances exploration and exploitation based on the posterior uncertainty (sigma) and the gap to the best (f_best - mu).
    - It avoids the pitfalls of normalizing scores to [0,1] which can distort the relative importance of candidates.
    """
    if time.time() > _DEADLINE:
        # Safety net: return a valid array if we are about to time out
        # Returning zeros is neutral, but we want to ensure we don't crash.
        # Ideally we should have returned earlier.
        return np.zeros_like(mu)

    # Ensure inputs are valid
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, np.inf)
    
    # If f_best is not finite, something is wrong, but let's try to handle it gracefully
    if not np.isfinite(f_best):
        return np.zeros_like(mu)

    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical overflow in exp/cdf
    # For z > 50, Phi(z) ~ 1, phi(z) ~ 0
    # For z < -50, Phi(z) ~ 0, phi(z) ~ 0
    z_clipped = np.clip(z, -50, 50)
    
    # Calculate EI components
    # Using scipy.stats.norm for precision
    cdf_vals = norm.cdf(z_clipped)
    pdf_vals = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    
    # Ensure EI is non-negative (numerical errors might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite outputs
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
