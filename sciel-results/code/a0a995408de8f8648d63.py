# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive kappa: decrease as we get closer to optimum
    if f_best > 10:
        kappa = 3.0
    elif f_best > 5:
        kappa = 2.5
    elif f_best > 2:
        kappa = 2.0
    elif f_best > 1:
        kappa = 1.5
    elif f_best > 0.5:
        kappa = 1.0
    else:
        kappa = 0.5
    
    # Lower Confidence Bound for minimization: mu - kappa * sigma
    # We want to maximize the score, so: -mu + kappa * sigma
    lcb_score = -mu + kappa * sigma_safe
    
    # Use percentile-based normalization for robustness
    ei_p5 = np.percentile(ei, 5)
    ei_p95 = np.percentile(ei, 95)
    ei_range = ei_p95 - ei_p5 + 1e-10
    ei_norm = (ei - ei_p5) / ei_range
    ei_norm = np.clip(ei_norm, 0, 1)
    
    lcb_p5 = np.percentile(lcb_score, 5)
    lcb_p95 = np.percentile(lcb_score, 95)
    lcb_range = lcb_p95 - lcb_p5 + 1e-10
    lcb_norm = (lcb_score - lcb_p5) / lcb_range
    lcb_norm = np.clip(lcb_norm, 0, 1)
    
    # Adaptive weight: more EI when close to optimum, more LCB when far
    if f_best > 5:
        w_ei = 0.5
        w_lcb = 0.5
    elif f_best > 1:
        w_ei = 0.6
        w_lcb = 0.4
    else:
        w_ei = 0.7
        w_lcb = 0.3
    
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
