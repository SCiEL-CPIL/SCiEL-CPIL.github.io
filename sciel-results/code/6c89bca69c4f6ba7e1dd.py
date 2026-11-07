import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI is generally more robust than LCB for multimodal functions like Michalewicz 2D,
    as it directly optimizes the expected reduction in the best value found.
    """
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the Z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive.
    # If mu > f_best, z is negative.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # To avoid numerical issues with very large/small z, we can use scipy.stats
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI is 0 if the mean is already better than f_best and sigma is small?
    # The formula handles this: if mu < f_best, (f_best - mu) is positive.
    # If mu > f_best, (f_best - mu) is negative, but Phi(z) is small, and sigma*phi(z) might compensate?
    # Actually, standard EI is defined as E[max(f_best - f(x), 0)].
    # The formula above is correct for this.
    
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure scores are finite
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
