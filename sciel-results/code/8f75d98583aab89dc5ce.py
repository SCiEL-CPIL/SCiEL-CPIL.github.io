import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with global budget penalty.
    
    Parameters:
    mu: array of shape (N,), mean of GP
    sigma: array of shape (N,), std of GP
    f_best: float, best observed value so far
    cost: array of shape (N,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,), acquisition scores (higher is better)
    """
    epsilon = 1e-9
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, epsilon)
    
    # Calculate standard Expected Improvement components
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Standard EI terms
    ei_exploit = (f_best - mu) * norm.cdf(z)
    ei_explore = sigma * norm.pdf(z)
    
    # EI calculation
    ei = ei_exploit + ei_explore  # Standard EI
    
    # Use mean sigma as a proxy for exploration potential / phase
    # Early on, sigma is high; later, sigma is low as GP fills in
    mean_sigma = np.mean(sigma)
    
    # Normalize mean_sigma to get a phase indicator
    # In 4D with length scale 0.2, initial sigma might be around 1.0 (standardized)
    # As GP fills, sigma decreases. Let's use a reference value.
    # If mean_sigma is high, we are early; if low, we are late.
    # Let's define phase = 1 - mean_sigma / (mean_sigma + 0.5)
    # This gives phase ~0 when mean_sigma is large, and phase ~1 when mean_sigma is small.
    phase = 1.0 - mean_sigma / (mean_sigma + 0.5)
    
    # Clamp phase to [0, 1]
    phase = np.clip(phase, 0.0, 1.0)
    
    # Lambda increases with phase (more exploitation later)
    # Also, we can use f_best as a secondary signal if needed, but let's stick to sigma.
    lambda_val = 0.5 + 2.0 * phase
    
    # Score = EI - lambda * cost
    scores = ei - lambda_val * cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
