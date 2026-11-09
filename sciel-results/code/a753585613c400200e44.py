# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill
_call_count = 0
_total_expected_calls = 25  # 30 total evals - 5 initial

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a 2D RBF mixture.
    
    Uses Expected Improvement (EI) with an adaptive exploration parameter (xi).
    
    Parameters:
        mu: array of shape (2000,), posterior mean at candidates
        sigma: array of shape (2000,), posterior std at candidates
        f_best: scalar, best (lowest) observed value so far
    
    Returns:
        array of shape (2000,), scores (higher = more promising to evaluate next)
    """
    global _call_count
    
    # Deadline guard: if we're close to the time limit, return a simple score quickly
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return something finite and simple
        return np.zeros_like(mu)
    
    # Increment call count to estimate progress
    _call_count += 1
    
    # Adaptive xi: start with higher exploration, decay to near 0
    # Linear decay from 0.1 to 0.01 over the expected 25 calls
    progress = min(_call_count / _total_expected_calls, 1.0)
    xi = 0.1 * (1.0 - progress) + 0.01 * progress
    
    # Ensure sigma is positive and finite to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement with xi
    # For minimization, we want mu < f_best - xi
    z = (f_best - mu - xi) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement for minimization with xi
    # EI_xi = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
