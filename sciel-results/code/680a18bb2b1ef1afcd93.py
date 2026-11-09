"""Cost-aware acquisition for three-hump-camel 2D BO using EI with cost penalty."""
import time
import numpy as np
from scipy.stats import norm

# Module-level cache for fast normal CDF/PDF computation
_CDF_CACHE = {}
_PDF_CACHE = {}


def _norm_cdf(z):
    """Fast normal CDF using lookup table with interpolation."""
    z = np.asarray(z, dtype=np.float64)
    # Clip to reasonable range
    z = np.clip(z, -10, 10)
    # Use scipy's norm.cdf for accuracy, it's fast enough
    return norm.cdf(z)


def _norm_pdf(z):
    """Fast normal PDF."""
    z = np.asarray(z, dtype=np.float64)
    return norm.pdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (lower is better)
    cost : ndarray, shape (2000,)
        Evaluation cost for each candidate point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)

    # Compute Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    cdf_z = _norm_cdf(z)
    pdf_z = _norm_pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost penalty: penalize expensive points
    # cost is in [1, 2] (1 + mean of coordinates scaled to [0,1])
    # Use absolute normalization for stability: cost_norm in [0, 1]
    cost_norm = (cost - 1.0) / 1.0  # [0, 1], higher = more expensive
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    
    # Adaptive weight based on how close f_best is to optimum (0)
    # f_best for three-hump-camel starts around 1-5, ends near 0
    convergence = max(0.0, min(1.0, 1.0 - f_best / 5.0))
    
    # Penalty strength increases as we converge
    alpha = 0.2 + 0.5 * convergence  # [0.2, 0.7]
    
    # Score: EI with exponential cost penalty (always positive, smooth)
    score = ei * np.exp(-alpha * cost_norm)
    
    # Add small exploration bonus for high uncertainty when not converged
    sigma_max = np.max(sigma) + 1e-10
    sigma_norm = sigma / sigma_max
    exploration_bonus = 0.03 * sigma_norm * (1.0 - convergence)
    
    score += exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() > deadline:
        # Return whatever we have, it's already computed
        pass
    
    return score
