"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-inspired acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a combination of Expected Improvement (EI) and Expected Squared Improvement (ESI)
    to capture the area-based nature of hypervolume improvement. The ESI term weights
    larger potential improvements more strongly, which is consistent with hypervolume
    being an area measure. This creates a non-linear weighting that favors candidates
    with both high expected improvement and high uncertainty, better balancing
    exploration and exploitation for multimodal landscapes like Branin 2D.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        The best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores, higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Compute z-score: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -20, 20)
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Ensure cdf and pdf are finite
    cdf = np.where(np.isfinite(cdf), cdf, 0.5)
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    
    # Expected Improvement (EI) - first moment of improvement
    m = f_best - mu
    ei = m * cdf + sigma * pdf
    
    # Expected Squared Improvement (ESI) - second moment
    # E[max(f_best - f, 0)^2] where f ~ N(mu, sigma^2)
    # For Y = f_best - f ~ N(m, sigma^2), E[Y^2 * 1{Y > 0}]
    # = m^2 * Phi(z) + sigma^2 * (1 - Phi(-z) + z*phi(z) - z*Phi(z))
    # Using Phi(z) = 1 - Phi(-z):
    # = m^2 * Phi(z) + sigma^2 * (Phi(z) + z*phi(z) - z*Phi(z))
    # = m^2 * Phi(z) + sigma^2 * (1 - z)*Phi(z) + sigma^2 * z*phi(z)
    # Simplified: m^2 * cdf + sigma^2 * (1 - z) * cdf + sigma^2 * z * pdf
    esi = m**2 * cdf + sigma**2 * (1 - z) * cdf + sigma**2 * z * pdf
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    esi = np.where(np.isfinite(esi), esi, 0.0)
    
    # Normalize ESI by a scale factor to make it comparable to EI
    # Use the median sigma as a robust scale estimate
    scale = np.median(sigma)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    
    # Normalize both terms to have similar magnitudes
    ei_norm = ei / (scale + 1e-9)
    esi_norm = esi / (scale**2 + 1e-9)
    
    # Combine EI and ESI with weights that emphasize the area-based nature of hypervolume
    # ESI is weighted more heavily to capture larger potential improvements
    # A small amount of EI provides stability
    score = 0.3 * ei_norm + 0.7 * esi_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
