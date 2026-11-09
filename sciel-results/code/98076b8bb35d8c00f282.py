import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 300s limit, start timer
    start_time = time.time()
    deadline = start_time + 295.0 # 5s safety margin

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost_safe
    
    # Dynamic exploration weight:
    # High f_best (bad) -> more exploration. Low f_best (good) -> less exploration.
    # Use a smoother transition centered around f_best = 2.0
    w = 1.0 / (1.0 + np.exp(-(f_best - 2.0) / 1.5))
    
    # Exploration term: use normalized sigma to prevent global uncertainty from dominating
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Add a small bonus for very promising points (mu well below f_best)
    # This helps when EI is small but the GP predicts a much better value
    mu_bonus = np.maximum(0, (f_best - mu) / (f_best + 1e-6))
    mu_bonus = mu_bonus / cost_safe * 0.1
    
    # Final score: balance EI, exploration, and promising-prediction bonus
    score = score_ei + 0.3 * w * sigma_norm + mu_bonus
    
    # Check time
    if time.time() > deadline:
         # Return best result found so far (which is just this calculation)
         return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
