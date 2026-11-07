import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using GP-LCB (Lower Confidence Bound) for minimization.
    Structural change: replaces EI-based formulation with LCB to avoid division
    by cost in the improvement term, providing a cleaner exploitation-exploration
    tradeoff via the beta hyperparameter.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # GP-LCB for minimization: lower bound on the function value
    # We want to minimize, so we pick points where the lower bound is lowest.
    # Higher score = more promising, so we negate the LCB.
    # LCB(x) = mu(x) - beta * sigma(x)
    # Score = -LCB(x) / cost(x) = (beta * sigma(x) - mu(x)) / cost(x)
    
    # Beta controls exploration: larger beta = more exploration
    beta = 2.0
    
    # LCB score: (beta * sigma - mu) / cost
    # This is high when mu is low (exploitation) or sigma is high (exploration)
    lcb_score = (beta * sigma - mu) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    return scores
