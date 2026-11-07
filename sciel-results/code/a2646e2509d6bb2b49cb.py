import numpy as np
from scipy.stats import t

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Student's t-distribution based Expected Improvement.
    
    This approach accounts for heavy-tailed uncertainty inherent in limited sample 
    budgets (30 evaluations for 6D Hartmann). The t-distribution prevents 
    over-confidence in the GP mean compared to the Gaussian assumption, promoting 
    more robust exploration.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Degrees of freedom for the t-distribution.
    # Lower degrees of freedom imply heavier tails and more exploration.
    # 5.0 is a reasonable choice for a small budget like 30 evaluations.
    nu = 5.0
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # CDF and PDF of the t-distribution
    cdf = t.cdf(z, df=nu)
    pdf = t.pdf(z, df=nu)
    
    # Expected Improvement for t-distribution
    # EI = (f_best - mu) * F_nu(z) + sigma * (nu / (nu - 1)) * (1 + z^2 / nu) * t_nu(z)
    # Note: The term (nu / (nu - 1)) * (1 + z^2 / nu) comes from the expectation of 
    # max(z, 0) under the t-distribution scaled appropriately.
    factor = (nu / (nu - 1.0)) * (1.0 + (z ** 2) / nu)
    
    ei = (f_best - mu) * cdf + sigma_safe * factor * pdf
    
    # Ensure finite scores to satisfy the contract
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
