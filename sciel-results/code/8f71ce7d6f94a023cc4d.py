import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function that switches between UCB and EI based on the current best value.
    """
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Upper Confidence Bound (UCB) for minimization
    # We maximize beta * sigma - mu
    beta = 1.5
    ucb = beta * sigma_safe - mu
    
    # Switch logic:
    # If f_best is still relatively high (far from the minimum ~0.4),
    # we prioritize exploration using UCB.
    # Once f_best is close to the minimum, we switch to exploitation using EI.
    threshold = 2.0
    
    if f_best > threshold:
        return ucb
    else:
        return ei
