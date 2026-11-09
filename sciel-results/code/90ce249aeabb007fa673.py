import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    EI is a well-established acquisition function that balances exploration and
    exploitation by computing the expected improvement over the current best.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The lowest observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher score indicates higher priority.
    """
    start_time = time.time()
    # Safety margin: 330s limit, stop at 320s
    deadline = start_time + 320.0
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    #
    # Closed form:
    # If sigma > 0:
    #   z = (f_best - mu) / sigma
    #   EI = sigma * (z * phi(z) + Phi(z))
    # where phi is the standard normal PDF and Phi is the standard normal CDF
    #
    # If sigma <= 0:
    #   EI = max(f_best - mu, 0)
    
    # Handle zero or near-zero sigma
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    # Compute z-score: how many standard deviations below f_best the mean is
    z = (f_best - mu) / safe_sigma
    
    # Compute EI using closed form
    # phi(z) = standard normal PDF
    # Phi(z) = standard normal CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = safe_sigma * (z * phi_z + Phi_z)
    
    # For points where sigma was originally ~0, use deterministic improvement
    low_sigma_mask = sigma < eps
    if np.any(low_sigma_mask):
        ei[low_sigma_mask] = np.maximum(f_best - mu[low_sigma_mask], 0.0)
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure all scores are finite and non-negative
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
