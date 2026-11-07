import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI and PI
    z = (f_best - mu) / sigma
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Determine weight for EI vs PI based on global uncertainty
    # High uncertainty -> Use PI (less exploratory, avoids over-exploration)
    # Low uncertainty -> Use EI (better for refinement)
    mean_sigma = np.mean(sigma)
    
    # Sigmoid switch
    # If mean_sigma is high, w is low (more PI)
    # If mean_sigma is low, w is high (more EI)
    threshold = 0.2
    temperature = 0.2
    w = 1.0 / (1.0 + np.exp((mean_sigma - threshold) / temperature))
    
    # Blend scores
    score = w * ei + (1.0 - w) * pi
    
    return score
