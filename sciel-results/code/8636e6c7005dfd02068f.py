import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x). UCB for minimization is mu - kappa * sigma.
    # We want to MAXIMIZE the score, so we use kappa * sigma - mu.
    # This balances exploitation (low mu) and exploration (high sigma).
    kappa = 2.5
    ucb = kappa * sigma_safe - mu
    
    # Divide by cost to prefer cheaper evaluations
    # Cost is in [1, 2]
    scores = ucb / cost
    
    # Handle any potential NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline (though this is fast)
    if time.time() > deadline:
         # Fallback: return random or uniform scores if somehow slow
         return np.ones_like(mu)
         
    return scores
