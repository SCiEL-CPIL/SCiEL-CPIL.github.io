import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a quantile-based
    exploration bonus for high-uncertainty, promising-mean candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # 2. Quantile-based exploration bonus
    # Top 5% uncertainty candidates that also have mu < f_best
    sigma_threshold = np.percentile(sigma, 95)
    high_uncertainty_mask = sigma >= sigma_threshold
    promising_mean_mask = mu < f_best
    exploration_mask = high_uncertainty_mask & promising_mean_mask
    
    # Scale bonus by the potential improvement range
    range_improvement = f_best - np.min(mu)
    if range_improvement < 1e-6:
        range_improvement = 1.0
    bonus_value = 0.1 * range_improvement
    
    scores = ei.copy()
    scores[exploration_mask] += bonus_value
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
