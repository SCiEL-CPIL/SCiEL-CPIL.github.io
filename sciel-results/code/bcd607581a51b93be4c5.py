import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # Lower is better, so we want to minimize mu - kappa * sigma
    # To get a score where higher is better, we use -mu + kappa * sigma
    kappa = 2.0
    ucb = -mu + kappa * sigma_safe
    
    # Normalize both EI and UCB to [0, 1] range for fair combination
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones_like(ei)
        
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb)
        
    # Combine EI and UCB with a balance parameter
    # Weight towards exploitation (EI) as we get closer to optimal, but keep some exploration
    alpha = 0.7  # 70% EI, 30% UCB
    scores = alpha * ei_norm + (1 - alpha) * ucb_norm
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback: Greedy (minimize mu)
        scores = -mu
        
    return scores
