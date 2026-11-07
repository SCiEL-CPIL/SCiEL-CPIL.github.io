"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t

# Global state is not strictly necessary for stateless acquisition, 
# but kept for potential future adaptive strategies if needed.
# For this implementation, the acquisition is stateless based on current posterior.

def acquisition(mu, sigma, f_best):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 300s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues in division and t-distribution parameters
    sigma = np.maximum(sigma, 1e-8)
    
    # Degrees of freedom for the Student's t-distribution.
    # Lower nu means heavier tails, promoting more exploration.
    # 5 is a common choice for balancing exploration and exploitation in BO.
    nu = 5.0
    
    # Compute z-score for the improvement
    # z = (f_best - mu) / sigma
    # Note: For minimization, we want f(x) < f_best.
    # The improvement is f_best - f(x).
    # If f(x) ~ N(mu, sigma^2), then (f_best - f(x)) ~ N(f_best - mu, sigma^2).
    # Standardized variable z = (f_best - mu) / sigma.
    z = (f_best - mu) / sigma
    
    # Compute Expected Improvement for Student's t-distribution
    # Formula: EI = sigma * [ z * F_t(z; nu) + (nu - 1) / (nu + 1) * f_t(z; nu) ]
    # where F_t is the CDF and f_t is the PDF of the standard t-distribution with nu degrees of freedom.
    
    # Use scipy.stats.t for numerical stability and accuracy
    # t.cdf(z, df)
    # t.pdf(z, df)
    
    F_t = t.cdf(z, nu)
    f_t = t.pdf(z, nu)
    
    # Coefficient for the PDF term
    coeff = (nu - 1.0) / (nu + 1.0)
    
    # Calculate EI
    ei = sigma * (z * F_t + coeff * f_t)
    
    # Check for non-finite values
    if not np.all(np.isfinite(ei)):
        ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=-1e10)
        
    # The t-distribution EI naturally provides more exploration than Gaussian EI due to heavier tails.
    # However, to maintain the benefit of explicit exploration found in previous iterations 
    # (which helped with the 6D Hartmann landscape), we can add a small exploration term.
    # Given the t-EI is already more exploratory, a smaller term than 0.1*sigma might be appropriate,
    # or we can omit it. Let's try a small term 0.05*sigma to be safe, or pure t-EI.
    # The previous best used 0.1*sigma with Gaussian EI.
    # Let's try pure t-EI first as the heavy tails should suffice.
    # If we want to be conservative and keep the "family" of EI + exploration, 
    # we can add a small term. 
    # Let's stick to pure t-EI to isolate the effect of the distribution change.
    
    scores = ei
    
    # Check deadline
    if time.time() > deadline:
        # The computation is O(N) and very fast, so this is unlikely to be hit.
        pass

    return scores
# EVOLVE-BLOCK-END
