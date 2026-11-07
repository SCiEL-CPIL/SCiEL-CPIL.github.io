import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Blend EI and PI
    # This structural change introduces a smoother exploration component
    # compared to pure EI, which can be too greedy or spiky.
    # Based on observed improvements in similar tasks.
    score = 0.5 * ei + 0.5 * pi
    
    return score
