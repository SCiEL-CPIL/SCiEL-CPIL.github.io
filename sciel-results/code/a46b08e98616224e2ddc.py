import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    # Normalize by cost to get expected improvement per unit cost.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Cost-aware: divide EI by cost to get improvement per unit cost
    # This makes the acquisition function explicitly cost-sensitive
    scores = ei / cost
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
