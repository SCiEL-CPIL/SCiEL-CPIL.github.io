import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Z-score: (f_best - mu) / sigma
    # For minimization, improvement is when mu < f_best
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(Z) - sigma * phi(Z) for Z > 0
    # EI = 0 for Z <= 0
    
    # Compute only where Z > 0 (potential improvement)
    Z_pos = np.where(Z > 0, Z, 0)
    
    # Phi(Z) = CDF of standard normal
    # phi(Z) = PDF of standard normal
    phi_Z = norm.pdf(Z_pos)
    Phi_Z = norm.cdf(Z_pos)
    
    # EI calculation
    EI = (f_best - mu) * Phi_Z - sigma_safe * phi_Z
    
    # Ensure non-negative (numerical safety)
    EI = np.maximum(EI, 0)
    
    return EI
