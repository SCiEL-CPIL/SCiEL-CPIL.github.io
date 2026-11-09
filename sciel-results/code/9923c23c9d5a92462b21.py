import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Expected Improvement (EI) with a small constant exploration component.
    This approach was found to outperform hybrid EI/UCB strategies for this task.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    # Use a very small epsilon to maintain numerical stability
    safe_sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    
    # EI calculation:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / safe_sigma
    
    # Standard normal CDF and PDF
    # Using scipy.stats.norm for accuracy
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    EI = (f_best - mu) * Phi + safe_sigma * phi
    
    # Add a small exploration term based on sigma.
    # A constant weight of 0.1 provides a good balance between exploitation
    # (via EI) and exploration (via sigma) without the instability of adaptive
    # weights or full UCB.
    exploration_weight = 0.1
    scores = EI + exploration_weight * safe_sigma
    
    # Ensure all scores are finite to satisfy the contract
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
