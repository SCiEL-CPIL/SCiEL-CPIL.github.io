"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as t_dist


def acquisition(mu, sigma, f_best):
    """
    Student's t-distribution based Expected Improvement acquisition function.
    
    Uses a t-distribution with nu=5 degrees of freedom to model the predictive
    uncertainty, which has heavier tails than the Gaussian. This accounts for
    the heavy-tailed uncertainty inherent in the limited 30-sample budget,
    preventing over-confidence in the GP mean and promoting more robust
    exploration of the 6D landscape.
    
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
    
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Degrees of freedom for the t-distribution
    nu = 5.0
    
    # Standardized variable: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # t-distribution CDF and PDF
    t_cdf = t_dist.cdf(z, df=nu)
    t_pdf = t_dist.pdf(z, df=nu)
    
    # Expected Improvement using t-distribution:
    # EI = (f_best - mu) * P(X < f_best) + sigma * pdf_t(z)
    # where P(X < f_best) = t_cdf(z) and pdf_t is the t-distribution pdf
    # scaled appropriately.
    
    # The standard EI formula with t-distribution:
    # EI = (f_best - mu) * t_cdf(z) + sigma * t_pdf(z)
    
    ei = (f_best - mu) * t_cdf + sigma_safe * t_pdf
    
    # Clip to finite values to ensure validity
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
