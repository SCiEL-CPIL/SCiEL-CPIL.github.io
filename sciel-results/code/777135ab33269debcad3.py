import numpy as np

def acquisition(mu, sigma, f_best):
       # Negative Lower Confidence Bound (NLCB)
       # For minimization, we want to maximize: beta * sigma - mu
       # This encourages points with low mean and high uncertainty.
       beta = 1.5
       # Ensure numerical stability
       sigma_safe = np.maximum(sigma, 1e-9)
       scores = beta * sigma_safe - mu
       return scores
