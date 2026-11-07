import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Time guard (optional but requested)
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Add epsilon to sigma to avoid division by zero
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # 1. Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # 3. Lower Confidence Bound (LCB) based score
    # We want to maximize (f_best - (mu - beta * sigma))
    # Score = f_best - mu + beta * sigma
    beta = 2.0
    lcb_score = (f_best - mu) + beta * sigma
    
    # Clip LCB score to 0, as negative scores indicate no improvement potential
    lcb_score = np.maximum(lcb_score, 0.0)
    
    # Calculate cost-adjusted scores
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    score_ei = ei / cost_safe
    score_pi = pi / cost_safe
    score_lcb = lcb_score / cost_safe
    
    # Normalize each score vector to [0, 1] to allow fair comparison
    def min_max_norm(x):
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val - min_val < 1e-12:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)
    
    norm_ei = min_max_norm(score_ei)
    norm_pi = min_max_norm(score_pi)
    norm_lcb = min_max_norm(score_lcb)
    
    # Combine by taking the maximum
    # This picks the candidate that is most promising according to ANY of the criteria
    final_score = np.maximum(np.maximum(norm_ei, norm_pi), norm_lcb)
    
    # Ensure finite values
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Time guard check (should be very fast)
    elapsed = time.time() - start_time
    if elapsed > 290: # Safety margin
         # If we somehow took too long, just return EI-based score (simplest)
         return score_ei
         
    return final_score
