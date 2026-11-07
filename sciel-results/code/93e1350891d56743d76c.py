import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    EI is a well-established acquisition function that balances exploration
    and exploitation. It naturally adapts: when f_best is far from the optimum,
    EI favors high-uncertainty regions; when close, it focuses on improving
    the best value.
    """
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF and PDF
    # Using scipy for accuracy
    from scipy.stats import norm
    
    # Z-score: (f_best - mu) / sigma
    # When mu < f_best, z is positive (improvement expected)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    # This is always non-negative
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite and non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
