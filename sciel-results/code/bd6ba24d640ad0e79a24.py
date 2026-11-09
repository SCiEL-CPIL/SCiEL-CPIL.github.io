"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()


def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a hybrid strategy:
    - Expected Improvement (EI) for exploitation
    - Lower Confidence Bound (LCB) for exploration
    - Dynamic blending based on f_best and uncertainty
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    # 10s safety margin.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    # For minimization: improvement is f_best - mu(x) when mu(x) < f_best
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF
    from math import sqrt, pi
    
    # Vectorized computation using numpy
    sqrt2 = np.sqrt(2.0)
    
    try:
        from scipy.stats import norm
        Phi_z = norm.cdf(z)
        phi_z = norm.pdf(z)
    except ImportError:
        # Manual computation
        # Phi(z) using error function
        Phi_z = 0.5 * (1.0 + np.erf(z / sqrt2))
        phi_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * pi)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI should be non-negative; clip any numerical issues
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound for exploration
    # Adaptive kappa: explore more when f_best is high, exploit more when low
    if f_best > 3.0:
        kappa = 2.5
    elif f_best > 1.0:
        kappa = 1.8
    else:
        kappa = 1.2
    
    lcb = -mu + kappa * sigma
    
    # Dynamic blending: use more EI when we have good estimates, more LCB when uncertain
    # Compute average uncertainty to determine blend weight
    avg_sigma = np.mean(sigma)
    
    # If uncertainty is high, lean more towards exploration (LCB)
    # If uncertainty is low, lean more towards exploitation (EI)
    if avg_sigma > 0.5:
        blend_weight = 0.3  # 30% EI, 70% LCB
    elif avg_sigma > 0.1:
        blend_weight = 0.6  # 60% EI, 40% LCB
    else:
        blend_weight = 0.9  # 90% EI, 10% LCB
    
    # Normalize both scores to have similar scale for blending
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    lcb_max = np.max(lcb) if np.max(lcb) > 0 else 1.0
    
    ei_norm = ei / ei_max
    lcb_norm = lcb / lcb_max
    
    # Blend the two acquisition functions
    score = blend_weight * ei_norm + (1 - blend_weight) * lcb_norm
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
