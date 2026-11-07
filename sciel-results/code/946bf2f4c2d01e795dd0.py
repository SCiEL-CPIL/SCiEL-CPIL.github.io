"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as t_dist


def acquisition(mu, sigma, f_best):
    """
    Student's t-distribution based Expected Improvement acquisition function.
    
    Uses the exact Expected Improvement formula for a Student's t-distribution
    with nu=5 degrees of freedom. The GP posterior standard deviation is mapped
    to the t-distribution scale parameter to ensure correct variance matching.
    This robustly handles the limited sample budget by accounting for
    heavy-tailed uncertainty.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP posterior mean at candidate points (in objective units).
    sigma : array-like, shape (n,)
        GP posterior standard deviation at candidate points (in objective units).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Degrees of freedom for the t-distribution
    nu = 5.0
    
    # Map GP standard deviation to t-distribution scale parameter
    # Var(t) = scale^2 * nu / (nu - 2)
    # sigma^2 = scale^2 * nu / (nu - 2) => scale = sigma * sqrt((nu - 2) / nu)
    scale = sigma_safe * np.sqrt((nu - 2.0) / nu)
    
    # Avoid division by zero in z calculation
    scale_safe = np.maximum(scale, 1e-9)
    
    # Standardized variable: z = (f_best - mu) / scale
    z = (f_best - mu) / scale_safe
    
    # t-distribution CDF and PDF
    t_cdf = t_dist.cdf(z, df=nu)
    t_pdf = t_dist.pdf(z, df=nu)
    
    # Exact Expected Improvement for Student's t-distribution:
    # EI = (f_best - mu) * T_nu(z) + scale * (nu / (nu - 1)) * t_pdf(z) * (1 + z^2 / nu)
    
    correction_factor = (nu / (nu - 1.0)) * (1.0 + (z ** 2) / nu)
    
    ei = (f_best - mu) * t_cdf + scale_safe * correction_factor * t_pdf
    
    # Clip to finite values to ensure validity
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
