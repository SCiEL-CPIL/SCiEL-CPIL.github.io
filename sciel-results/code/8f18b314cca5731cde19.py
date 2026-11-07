import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI is generally superior to UCB for minimization in low dimensions
    with limited budgets because it accounts for the probability of improvement.
    The previous program used EI + a heuristic exploration term. 
    We hypothesize that the heuristic term introduces noise that degrades performance,
    and that the natural exploration properties of EI (via the prior mean and uncertainty)
    are sufficient when the GP prior is well-calibrated (standardized targets).
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Z-score for Expected Improvement
    Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in CDF/PDF
    Z = np.clip(Z, -10, 10)
    
    # Phi(Z) = standard normal CDF, phi(Z) = standard normal PDF
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Ensure finite scores
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
