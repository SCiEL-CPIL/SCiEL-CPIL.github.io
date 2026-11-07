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
    Acquisition function for Bayesian optimization.
    
    Uses Variance-Gated Expected Improvement (VEI): scales standard EI by
    (sigma / sigma_max)^gamma to explicitly penalize exploitation in
    low-uncertainty regions and force exploration where the GP is still
    uncertain. This is critical for escaping local minima in Branin's
    multimodal landscape.
    
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
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure sigma is positive to avoid division issues
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Variance-Gated EI: scale by (sigma / sigma_max)^gamma
    sigma_max = np.max(sigma_safe)
    if sigma_max < eps:
        # All sigmas are essentially zero; fall back to pure EI
        scores = ei
    else:
        gamma = 1.5
        sigma_ratio = sigma_safe / (sigma_max + eps)
        gate = np.power(sigma_ratio, gamma)
        
        # VEI = EI * gate
        # This penalizes exploitation in low-uncertainty regions
        scores = ei * gate
    
    # Add a small PI term to help differentiate among candidates
    # with similar VEI but different uncertainty
    pi = cdf_z
    alpha = 0.02  # Small weight on PI for additional exploration signal
    scores = scores + alpha * pi
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
