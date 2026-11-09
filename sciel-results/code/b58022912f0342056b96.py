import numpy as np
from scipy.special import erf, exp

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF using scipy.special for stability and speed
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    cdf_z = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # phi(z) = 1/sqrt(2*pi) * exp(-z^2/2)
    pdf_z = (1.0 / np.sqrt(2.0 * np.pi)) * exp(-0.5 * z**2)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps in multimodal functions by encouraging exploration of high-variance regions
    exploration_bonus = 0.1 * sigma_safe
    
    total_improvement = ei + exploration_bonus
    
    # Cost-aware score
    score = total_improvement / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
