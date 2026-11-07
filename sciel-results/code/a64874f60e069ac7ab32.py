import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle sigma == 0 to avoid division by zero
    # If sigma is very small, use the deterministic improvement
    eps = 1e-10
    sigma_safe = np.where(sigma < eps, eps, sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # CDF and PDF of standard normal
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI formula
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # If sigma was effectively 0, EI should be max(0, f_best - mu)
    # The formula above with sigma_safe=eps approximates this well if f_best > mu, 
    # but let's be precise for the sigma==0 case to avoid numerical artifacts.
    # Actually, if sigma=0, z is huge. 
    # If mu < f_best, z is huge positive. cdf->1, pdf->0. ei -> f_best - mu.
    # If mu > f_best, z is huge negative. cdf->0, pdf->0. ei -> 0.
    # So the formula handles it naturally if we don't divide by zero.
    
    # Ensure non-negative scores (EI is always non-negative)
    # Higher EI is better
    scores = np.maximum(ei, 0.0)
    
    # Handle any NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
