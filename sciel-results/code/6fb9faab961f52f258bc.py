import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization (styblinski tang 2d).
    
    Uses a pure cost-normalized Lower Confidence Bound:
    score(x) = -(mu(x) - kappa * sigma(x)) / cost(x)
    
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
        # Fallback: simple LCB
        kappa = 2.0
        scores = (-mu + kappa * sigma) / cost
        return np.where(np.isfinite(scores), scores, -1e10)
    
    # Structural Change: Cost-Normalized Lower Confidence Bound (LCB)
    # We want to minimize mu - kappa * sigma.
    # To maximize score (higher is better), we use:
    # score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    
    kappa = 1.0
    
    # Calculate LCB score
    scores = (kappa * sigma - mu) / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
