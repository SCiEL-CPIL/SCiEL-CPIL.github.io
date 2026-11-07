import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - Y(x), 0)]
    # Assuming Y(x) ~ N(mu, sigma^2)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.where(sigma < 1e-10, 1.0, sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # If sigma was effectively 0, EI should be max(0, f_best - mu)
    # The formula above with sigma=0 becomes:
    # (f_best - mu) * cdf(inf or -inf) + 0
    # If f_best > mu, z -> inf, cdf -> 1, ei -> f_best - mu. Correct.
    # If f_best < mu, z -> -inf, cdf -> 0, ei -> 0. Correct.
    # If f_best == mu, z -> 0, cdf -> 0.5, ei -> 0.5 * 0 + 0 = 0. Correct.
    
    # Cost-aware score: EI / cost
    scores = ei / cost
    
    return scores
