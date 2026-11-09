import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Deadline check
    def time_left():
        return 330.0 - (time.time() - t0) - 5.0
    
    if time_left() <= 0:
        # Fallback to simple LCB
        kappa = 2.0
        scores = (-mu + kappa * sigma) / cost
        return np.where(np.isfinite(scores), scores, -1e10)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    Z = (f_best - mu) / sigma
    
    # Calculate Phi(Z) and phi(Z)
    # Use vectorized operations
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score
    scores = EI / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
