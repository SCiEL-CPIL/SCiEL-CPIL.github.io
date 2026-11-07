import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Normalize EI and Sigma
    max_ei = np.max(ei)
    max_sigma = np.max(sigma)
    
    if max_ei < 1e-9:
        norm_ei = np.zeros_like(ei)
    else:
        norm_ei = ei / max_ei
        
    if max_sigma < 1e-9:
        norm_sigma = np.zeros_like(sigma)
    else:
        norm_sigma = sigma / max_sigma
        
    # Combine
    # Weights can be tuned, but 0.5/0.5 is a neutral start.
    # To favor exploration slightly (as EI is often too greedy), maybe 0.4/0.6?
    # But "do not micro-tune". 0.5/0.5 is structural.
    score = 0.5 * norm_ei + 0.5 * norm_sigma
    
    return score
