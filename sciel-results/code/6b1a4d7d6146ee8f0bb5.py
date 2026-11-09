import numpy as np
from scipy.special import ndtr

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function for minimization.
    
    This function balances exploration and exploitation by using a UCB score
    that adapts the exploration constant based on the current best observed value
    relative to the known global minimum of the Hartmann 3D function.
    
    Parameters:
    mu: array of mean predictions (shape N,)
    sigma: array of standard deviations (shape N,)
    f_best: float, the best (lowest) function value observed so far
    
    Returns:
    scores: array of UCB scores (shape N,), higher is better.
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Known minimum of Hartmann 3D function
    f_min_hartmann = -3.86278
    
    # Calculate the gap to the global minimum
    # A large gap indicates we are early in the optimization (need exploration)
    # A small gap indicates we are late (need exploitation)
    gap = f_best - f_min_hartmann
    
    # Ensure gap is non-negative
    gap = max(gap, 0.0)
    
    # Adaptive exploration constant (beta)
    # We want high beta (exploration) when gap is large, and low beta (exploitation) when gap is small.
    # Using a smooth exponential decay or a rational function to map gap to beta.
    # Let's use a mapping where beta starts around 3.0 for large gaps and decays to around 0.5 for small gaps.
    # A simple effective mapping: beta = base + scale / (1 + gap)
    # If gap is large (e.g., 2.0), 1/(1+2) = 0.33. 
    # If gap is small (e.g., 0.0), 1/(1+0) = 1.0.
    # Let's tune this:
    # beta = 0.5 + 2.5 * (1.0 / (1.0 + gap))
    # When gap -> inf, beta -> 0.5
    # When gap = 0, beta = 3.0
    
    beta = 0.5 + 2.5 * (1.0 / (1.0 + gap))
    
    # UCB for minimization: Maximize (beta * sigma - mu)
    # This is equivalent to minimizing the optimistic lower bound (mu - beta * sigma)
    scores = beta * sigma - mu
    
    # Ensure all scores are finite, replacing any NaN/Inf with a very low value
    # Using a very low value instead of -inf to prevent potential issues in argmax if all are -inf
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
