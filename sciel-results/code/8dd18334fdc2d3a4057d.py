import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Clip sigma to avoid division by zero
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Calculate the z-score for the improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF of the standard normal distribution
    # And phi is the PDF of the standard normal distribution
    
    # Use scipy.stats.norm for accuracy
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0)
    
    # Cost-aware: divide by cost
    # The task states: "Evaluating x costs 1 + mean(x scaled to [0, 1])"
    # The 'cost' array provided in the function signature corresponds to this.
    cost_safe = np.clip(cost, 1e-8, None)
    
    scores = ei / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
