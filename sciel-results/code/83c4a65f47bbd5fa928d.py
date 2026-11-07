import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by evaluation cost.
    
    Parameters:
    mu: np.ndarray, shape (2000,), GP mean predictions.
    sigma: np.ndarray, shape (2000,), GP standard deviation predictions.
    f_best: float, best observed function value.
    cost: np.ndarray, shape (2000,), cost of evaluating each candidate.
    
    Returns:
    scores: np.ndarray, shape (2000,), acquisition scores.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for improvement
    # z = (f_best - mu) / sigma
    # We want to find points where mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
