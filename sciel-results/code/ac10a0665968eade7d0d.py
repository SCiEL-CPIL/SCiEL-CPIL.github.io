import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Optimistic LCB with cost normalization.
    
    Structurally different from EI: uses a point estimate (mu - kappa*sigma)
    rather than an expectation over the posterior.
    """
    kappa = 2.0
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB for minimization (lower is better)
    lcb = mu - kappa * sigma_safe
    
    # Optimistic improvement over f_best
    improvement = np.maximum(f_best - lcb, 0)
    
    # Cost-normalized LCB score
    lcb_score = improvement / cost_safe
    
    # Probability of Improvement for balance
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    poi_score = poi / cost_safe
    
    # Combine: primarily LCB, with some PoI for stability
    score = 0.85 * lcb_score + 0.15 * poi_score
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
