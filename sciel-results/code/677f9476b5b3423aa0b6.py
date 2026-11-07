import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Deterministic Expected Improvement (EI) acquisition function.
    
    Strategy:
    Instead of stochastic Thompson Sampling, we compute the Expected Improvement
    for each candidate point. EI is a deterministic score that balances
    exploitation (high probability of improvement) and exploration (high
    uncertainty).
    
    For minimization:
    Let Z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    Where:
    - Phi is the standard normal CDF
    - phi is the standard normal PDF
    
    If sigma is very small, we handle it carefully to avoid division by zero.
    If sigma is 0, EI is simply max(0, f_best - mu).
    
    We return the EI values as scores. Higher EI means higher priority.
    """
    # Ensure sigma is non-negative and handle numerical stability
    # A small epsilon prevents division by zero and log issues
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # This represents how many standard deviations the current mean is below f_best
    z = (f_best - mu) / sigma_safe
    
    # Compute EI components
    # Phi(z) = CDF of standard normal
    # phi(z) = PDF of standard normal
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI formula for minimization
    # Term 1: Expected improvement if we sample from N(mu, sigma)
    # Term 2: Uncertainty bonus
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure finite values. If any calculation results in NaN or Inf, replace with 0.
    # This should rarely happen with stable inputs, but it's a safety net.
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is always non-negative.
    # We return EI as the score. The evaluator picks the argmax.
    return ei
