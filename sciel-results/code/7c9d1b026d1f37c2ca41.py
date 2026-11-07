import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for Bayesian optimization.
    
    For a single-step decision, the Knowledge Gradient is exactly equal to Expected Improvement (EI):
    KG(x) = E[ f_best - min(f_best, Y_x) ] = EI(x)
    
    For a Gaussian posterior Y_x ~ N(mu_x, sigma_x^2), EI has a closed form:
    EI(x) = (f_best - mu_x) * Phi((f_best - mu_x) / sigma_x) + sigma_x * phi((f_best - mu_x) / sigma_x)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    This is the theoretically optimal 1-step acquisition for minimization.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: how many sigma's is f_best above mu
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clamp to non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    return ei
