"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - _SAFETY_MARGIN:
        # Return a simple greedy score if we're running out of time
        return -mu
    
    # Expected Improvement (EI) acquisition function
    # For minimization:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # np.maximum(z, 0) ensures we only get positive values for points where mu < f_best
    # But actually, the formula is valid for all z, but EI is 0 when z <= 0 (i.e., mu >= f_best)
    # Let's use the standard formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But we need to be careful: if mu >= f_best, EI should be 0.
    
    # Using scipy for normal CDF and PDF
    from scipy.stats import norm
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # EI formula for minimization
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (due to numerical issues, small negative values might occur)
    ei = np.maximum(ei, 0.0)
    
    # Higher EI means more likely to improve, so higher score is better
    scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
