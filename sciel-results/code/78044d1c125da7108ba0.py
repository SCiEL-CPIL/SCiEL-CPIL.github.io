"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Deterministic Expected Improvement (EI) acquisition function with an
    adaptive exploration parameter (xi) that scales with the current posterior
    uncertainty.
    
    The key structural idea: use standard EI but with xi that adapts to the
    mean posterior standard deviation. When uncertainty is high (early in the
    search or in poorly-explored regions), xi is larger, shifting the
    improvement threshold and encouraging exploration. When uncertainty is
    low (late in the search, well-explored regions), xi shrinks toward zero,
    focusing on exploitation.
    
    This is structurally different from:
    - Fixed-parameter EI (which has a constant xi)
    - UCB/LCB (which uses a linear combination of mu and sigma)
    - Thompson sampling (which is stochastic)
    
    The adaptive xi leverages the GP's own uncertainty estimate to dynamically
    balance exploration and exploitation without fixed hyperparameters.
    
    For Michalewicz 2D, which has a narrow global minimum in a multimodal
    landscape, this helps:
    1. Early: high sigma -> high xi -> more exploration to find promising basins
    2. Late: low sigma -> low xi -> precise exploitation near the best point
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Pure exploitation fallback: pick the point with lowest mean
        return -mu
    
    n = len(mu)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute the mean posterior uncertainty as a proxy for overall exploration level
    sigma_mean = np.mean(sigma_safe)
    
    # Adaptive xi: scales linearly with the mean posterior uncertainty
    # When sigma_mean is large (early search), xi is larger -> more exploration
    # When sigma_mean is small (late search), xi is smaller -> more exploitation
    # The factor 0.1 is chosen to give reasonable exploration pressure
    # without being so aggressive that we ignore clearly better regions
    xi = 0.1 * sigma_mean
    
    # Ensure xi is non-negative
    xi = max(xi, 0.0)
    
    # Compute the adjusted improvement threshold
    # Standard EI uses f_best; adaptive EI uses f_best + xi
    # This shifts the "improvement" baseline upward, making it harder to
    # claim improvement and thus encouraging exploration of uncertain regions
    f_adjusted = f_best + xi
    
    # Compute z-score for the adjusted EI
    z = (f_adjusted - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # avoid numerical issues with extreme values
    
    # Expected Improvement with adaptive xi
    # EI = (f_adjusted - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_adjusted - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
