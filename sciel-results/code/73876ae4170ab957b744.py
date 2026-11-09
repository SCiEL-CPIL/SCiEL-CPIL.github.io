import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses a hybrid of Expected Improvement (EI) and a Lower Confidence Bound (LCB)
    based exploration term. The LCB term is formulated to be high when
    mu is low and sigma is high, which is desirable for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean of shape (2000,)
    sigma (np.ndarray): Posterior standard deviation of shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores of shape (2000,), higher is better.
    """
    # Deadline guard (though this function is fast, it's a safety measure)
    # We don't have a global start time, but we can assume the call is fast.
    # If this were a loop, we would check time here.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI)
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Exploration Term (LCB-based)
    # For minimization, we want to select points with low mu and high sigma.
    # LCB value = mu - beta * sigma (low is good).
    # To use as a "higher is better" score, we invert it: -LCB = beta * sigma - mu.
    beta = 2.5
    exploration_score = beta * sigma_safe - mu
    
    # 3. Hybrid Score
    # Combine EI (exploitation) and exploration_score (exploration + exploitation on mu)
    # Weighting: EI is often more sensitive to local improvements.
    # The exploration term helps escape local minima.
    score = ei + 0.5 * exploration_score
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
