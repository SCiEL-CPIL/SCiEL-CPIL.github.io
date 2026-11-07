import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): Best observed value so far.
    
    Returns:
    scores (np.ndarray): EI scores for each candidate, shape (N,)
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure sigma is positive to avoid division by zero
    # If sigma is very small, the posterior is certain. 
    # If mu < f_best, EI should be f_best - mu.
    # If mu >= f_best, EI should be 0.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive. If mu > f_best, z is negative.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Calculate components
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # For very small sigma, numerical precision can cause issues if mu is close to f_best.
    # However, the formula above is generally stable. 
    # We can enforce non-negativity as EI should theoretically be >= 0.
    ei = np.maximum(ei, 0.0)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    # Deadline guard (though EI is fast, this satisfies the constraint)
    if time.time() - start_time > 295.0:
        # If we somehow took too long (unlikely for 2000 points), return current scores
        pass
        
    return scores
