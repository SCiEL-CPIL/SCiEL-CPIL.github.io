import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses Expected Improvement (EI) with an adaptive exploration term.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) for exploration
    # We want to explore regions with high uncertainty
    # beta adapts based on how close we are to the optimum
    # When f_best is large (far from optimum), explore more
    # When f_best is small (close to optimum), exploit more
    beta = 0.5 + 1.5 * np.exp(-f_best / 5.0)
    lcb = -mu + beta * sigma
    
    # Combine EI and LCB
    # Normalize both to [0, 1] range for fair combination
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-9:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-9:
        lcb_norm = (lcb - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb)
    
    # Weight combination: more weight to EI when close to optimum
    # When f_best is small, rely more on EI; when large, rely more on LCB
    w_ei = 0.7 + 0.3 * np.exp(-f_best / 10.0)
    w_lcb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Safety check: ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        # Fallback to pure EI if something went wrong
        scores = ei.copy()
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        # Return best found so far with safety margin
        pass
    
    return scores
