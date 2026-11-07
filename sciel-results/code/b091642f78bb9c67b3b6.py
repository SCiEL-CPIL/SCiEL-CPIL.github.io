"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for potential timeout guards
_start_time = time.time()
_DEADLINE = _start_time + 290  # Keep a 10s safety margin before the 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Variance-Gated Expected Improvement (VEI).
    
    Scales standard EI by (sigma / sigma_max)^gamma to explicitly penalize
    exploitation in low-uncertainty regions, forcing exploration of areas
    where the GP is still uncertain.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return uniform scores as fallback
        return np.ones_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate standard EI for minimization:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = improvement * cdf + sigma_safe * pdf
    
    # Variance-gating factor: (sigma / sigma_max)^gamma
    # This penalizes low-uncertainty regions, encouraging exploration
    sigma_max = np.max(sigma_safe)
    if sigma_max < 1e-10:
        # All sigmas are essentially zero; return EI directly
        return ei
    
    gamma = 1.5
    variance_gate = (sigma_safe / sigma_max) ** gamma
    
    # Combine: VEI = EI * variance_gate
    # This structurally shifts the tradeoff: in low-sigma regions, even if EI
    # is high, the score is dampened; in high-sigma regions, the score is boosted
    scores = ei * variance_gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
