import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement (POI) divided by cost.
    Structural change: replaces EI-based formulation with POI/cost, which prioritizes
    points with high likelihood of beating current best at lowest cost, ignoring
    magnitude of improvement in favor of certainty.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Probability of Improvement (POI) for minimization
    # P(f(x) < f_best) = P(Z < (f_best - mu) / sigma) = norm.cdf((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Cost-Effective POI: probability of improvement per unit cost
    ce_poi = poi / cost
    
    # Ensure finite
    ce_poi = np.where(np.isfinite(ce_poi), ce_poi, 0.0)
    
    return ce_poi
