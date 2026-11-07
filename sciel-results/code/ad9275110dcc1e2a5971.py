"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive hedge parameter
    # A small fraction of the median uncertainty helps balance exploration and exploitation
    med_sigma = np.median(sigma_safe)
    xi = 0.01 * med_sigma
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu - xi) / sigma_safe
    
    # Calculate CDF and PDF using erf for better numerical stability and speed
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # phi(z) = (1 / sqrt(2 * pi)) * exp(-z^2 / 2)
    # Clip z^2 to avoid overflow in exp (though underflow to 0 is desired and safe)
    # exp(-large_number) -> 0, which is correct for PDF tails
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)
    
    # Expected Improvement
    ei = (f_best - mu - xi) * cdf + sigma_safe * pdf
    
    # Cost-aware score: EI per unit cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    score = ei / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
