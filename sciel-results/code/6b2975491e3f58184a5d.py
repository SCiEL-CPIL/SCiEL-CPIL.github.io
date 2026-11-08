import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * P(f(x) < f_best) + E[f_best - f(x) | f(x) < f_best] * P(f(x) < f_best)
    # E[f_best - f(x) | f(x) < f_best] = sigma * pdf(z) / cdf(z)  (if cdf > 0)
    # So EI = (f_best - mu) * cdf(z) + sigma * pdf(z)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Cost-aware scoring
    # Using sqrt(cost) to reduce the penalty for high-cost points compared to linear cost
    # This balances the trade-off between finding good points and avoiding expensive ones
    # without being too biased towards the cheapest (and potentially worst) regions.
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    scores = ei / np.sqrt(cost)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
