import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    This strategy uses the exact Expected Improvement formula, which is scale-invariant
    when mu and sigma are in objective units. It balances exploitation (low mu) and
    exploration (high sigma) more effectively than linear UCB by accounting for the
    full distribution of potential improvements.
    
    Parameters:
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (2000,), acquisition values (higher is better)
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Handle edge case where f_best is 0 or very small
    # If f_best is 0, we are at the minimum. EI should be 0 everywhere if mu >= 0.
    # However, due to noise or numerical issues, mu might be slightly negative.
    # We calculate EI as E[max(f_best - f(x), 0)].
    
    # Calculate Z = (f_best - mu) / sigma
    # This represents how many standard deviations the current best is above the mean.
    # If Z is large positive, mu is much lower than f_best, so improvement is likely.
    # If Z is large negative, mu is higher than f_best, so improvement is unlikely.
    
    # Avoid division by zero, though sigma_safe handles most cases.
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    phi_cdf = norm.cdf(z)
    
    # Compute EI
    # Note: (f_best - mu) can be negative, but the formula handles it correctly.
    # If f_best - mu is negative, Phi(z) is small, and the term is small.
    # The sigma * phi(z) term is always positive, representing the value of uncertainty.
    
    ei = (f_best - mu) * phi_cdf + sigma_safe * phi_z
    
    # Ensure finite scores
    # EI should be non-negative. Due to floating point errors, it might be slightly negative.
    # We clip to 0 to ensure non-negative scores, which is theoretically correct for EI.
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    # If all scores are 0 (shouldn't happen unless f_best is very low and mu is high),
    # we might want to return a small constant or sigma to encourage exploration.
    # However, returning 0 is valid and means "no preference".
    # To be safe and ensure some exploration if stuck, we can add a tiny epsilon * sigma
    # but pure EI is generally robust. Let's stick to pure EI for structural difference.
    
    return scores
