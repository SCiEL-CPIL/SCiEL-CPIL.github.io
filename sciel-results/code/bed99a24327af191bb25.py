"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu : array-like of shape (n,)
        Mean of the Gaussian process posterior.
    sigma : array-like of shape (n,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        The best (lowest) function value observed so far.
        
    Returns:
    array of shape (n,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle sigma close to zero to avoid division by zero
    # If sigma is very small, the distribution is a point mass.
    # EI is max(0, f_best - mu) in that case.
    eps = 1e-8
    sigma_safe = np.where(sigma < eps, eps, sigma)
    
    # Z-score: how many standard deviations f_best is above the mean
    # For minimization, we care about f_best - mu
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best < mu, z is negative, Phi(z) is small, but (f_best-mu) is negative.
    # The formula naturally handles this, resulting in a small positive value due to the uncertainty term,
    # or close to 0 if we are far away and certain.
    # Actually, strictly speaking, EI is 0 if we are certain the value is worse than f_best.
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) is valid for all z.
    # If mu > f_best + 3*sigma, z is large positive? No, z = (f_best - mu)/sigma.
    # If mu > f_best, f_best - mu < 0, so z < 0.
    # Let's check limits.
    # If mu >> f_best, z -> -inf. Phi(-inf)=0, phi(-inf)=0. EI -> 0. Correct.
    # If mu << f_best, z -> +inf. Phi(inf)=1, phi(inf)=0. EI -> f_best - mu. Correct.
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure non-negative (numerical noise might cause tiny negatives)
    ei = np.maximum(ei, 0)
    
    # If sigma was actually 0 (or < eps), the EI should be exactly max(0, f_best - mu)
    # The formula with eps might give a tiny non-zero value for points where sigma=0 and mu > f_best.
    # Let's mask those cases explicitly to be clean.
    zero_sigma_mask = sigma < eps
    if np.any(zero_sigma_mask):
        ei[zero_sigma_mask] = np.maximum(f_best - mu[zero_sigma_mask], 0)
        
    return ei
# EVOLVE-BLOCK-END
