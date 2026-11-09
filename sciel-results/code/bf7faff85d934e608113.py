import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For Gaussian posterior: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi and phi are the CDF and PDF of the standard normal distribution
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural Change: Hybrid EI + Exploration
    # Add an exploration bonus proportional to sigma to encourage exploring
    # regions with high uncertainty, balancing the exploitation of EI.
    # This hybrid approach is known to improve performance on multimodal functions
    # like Ackley by preventing premature convergence.
    scores = ei + 0.5 * sigma_safe
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback: Greedy (minimize mu)
        scores = -mu
        
    return scores
