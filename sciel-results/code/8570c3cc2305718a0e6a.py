import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute the Expected Improvement (EI) for minimizing the function:
       EI(x) = E[max(f_best - f(x), 0)]
    2. Adjust for cost: Since we want to maximize the "value" of the evaluation,
       and cost is a resource constraint, we scale EI by the inverse of cost.
       This creates a "cost-aware EI" or "expected improvement per unit cost".
       Score = EI(x) / cost(x)
    3. This is deterministic, unlike Thompson Sampling, which avoids the variance
       of random samples and directly optimizes the expected gain.
    
    Mathematical formulation:
    Let Z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    We handle the case where sigma is very small by clamping it.
    """
    
    # Ensure sigma is positive and not too small to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-6)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-6)
    
    # Compute Z-score for the improvement
    # We are minimizing, so improvement is f_best - f(x)
    # If mu is already below f_best, EI should be high (exploitation)
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma
    
    # Compute EI
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: If Z is large positive, Phi(Z) ~ 1, phi(Z) ~ 0, EI ~ f_best - mu
    # If Z is large negative, Phi(Z) ~ 0, phi(Z) ~ 0, EI ~ 0
    # This correctly handles both exploitation (low mu) and exploration (high sigma)
    
    # Use scipy.stats.norm for stable evaluation of Phi and phi
    # Phi(Z) = norm.cdf(Z)
    # phi(Z) = norm.pdf(Z)
    
    # To avoid underflow/overflow issues for extreme Z, we can use log-space or just rely on scipy's stability
    # scipy.stats.norm.cdf and pdf are generally stable.
    
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # Ensure EI is non-negative (numerical errors might cause tiny negatives)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: Maximize EI per unit cost
    # Higher score = better candidate
    scores = EI / cost
    
    # Handle non-finite values
    scores[~np.isfinite(scores)] = -1e9
    
    return scores
