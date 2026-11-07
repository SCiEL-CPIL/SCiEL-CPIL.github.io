import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds before the 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure inputs are numpy arrays for consistent operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety margins for division
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # 1. Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB) score
    # For minimization, LCB = mu - beta * sigma.
    # We want to maximize the acquisition score, so Score_LCB = -LCB = -mu + beta * sigma.
    # This encourages exploration (high sigma) and exploitation (low mu).
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # 3. Combine EI and LCB
    # Use maximum to allow switching between exploitation (EI) and exploration (LCB)
    # We can also use a weighted sum or soft-max, but max is robust.
    combined_score = np.maximum(ei, lcb_score)
    
    # 4. Cost-aware normalization
    score = combined_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
