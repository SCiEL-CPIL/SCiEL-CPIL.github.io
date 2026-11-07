"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timing guard
_ACQ_START = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s wall-clock limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Variance-Gated Expected Improvement (VEI).
    Standard EI is scaled by (sigma / sigma_max)^gamma to penalize
    exploitation in low-uncertainty regions and encourage exploration
    in areas where the GP is still uncertain. This helps escape local
    minima in the multimodal Branin landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed objective value so far (lower is better).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    # Check time budget
    elapsed = time.time() - _ACQ_START
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback: pick the point with lowest mu
        mu_arr = np.asarray(mu)
        # Fallback: use negative mu (lower mu is better)
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Variance-Gating: scale EI by (sigma / sigma_max)^gamma
    # This structurally penalizes exploitation in low-uncertainty regions
    sigma_max = np.max(sigma_safe)
    if sigma_max < eps:
        # If all sigmas are essentially zero, fall back to pure EI
        gate = np.ones_like(ei)
    else:
        sigma_ratio = sigma_safe / sigma_max
        gamma = 1.5
        gate = np.power(sigma_ratio, gamma)
    
    # Combine: VEI = EI * gate
    scores = ei * gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
