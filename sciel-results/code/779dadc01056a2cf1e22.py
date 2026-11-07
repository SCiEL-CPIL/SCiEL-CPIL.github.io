import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero or negative sigma (though sigma should be >= 0)
    # Clip sigma to a small positive value to prevent numerical instability
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # This represents how many standard deviations f_best is above the mean.
    # If mu < f_best, Z > 0.
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Phi is CDF of standard normal
    # phi is PDF of standard normal
    
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure EI is non-negative (numerical errors might cause tiny negatives)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: EI / cost
    # We want to maximize this score.
    # If cost is 0, this would be inf, but cost is 1 + mean(x_scaled) >= 1.
    score = EI / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
