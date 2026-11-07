import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Time guard
    if time.time() - _start_time > _DEADLINE:
        return np.random.rand(len(mu)) * 1e-6 + 1e-9

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    n = len(mu)
    
    # Ensure cost is safe for division
    cost_safe = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive for stability, though GP should output positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural Change: Pure Cost-Aware UCB with aggressive exploration
    # For small budgets (approx 20 evals) and multimodal functions (Griewank),
    # greedy EI is often too exploitative. UCB with a higher kappa 
    # encourages visiting under-explored regions which is crucial 
    # for finding the global minimum among many local ones.
    
    # Dynamic kappa: Start high, decrease slightly? 
    # We don't have access to the iteration count directly, 
    # but we can infer it from f_best? No.
    # Let's use a fixed, higher kappa than the previous 1.5.
    # 2.5 is a strong exploration parameter.
    kappa = 2.5
    
    # UCB for minimization: we want to maximize -mu + kappa * sigma
    # This score is high when mu is low (good) or sigma is high (uncertain).
    ucb_score = -mu + kappa * sigma_safe
    
    # Normalize by cost to account for the cost model
    # Cost is in [1, 2].
    scores = ucb_score / cost_safe
    
    # Add a small amount of random noise to break ties and encourage diversity
    # This helps if the GP surface is flat in some regions
    noise = np.random.normal(0, 0.01 * np.std(ucb_score), size=n)
    scores += noise
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
