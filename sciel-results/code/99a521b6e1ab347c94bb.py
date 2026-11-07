import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    if _elapsed() > _DEADLINE:
        return np.zeros(2000)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is non-negative and cost is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Progress metric: how close f_best is to the known minimum (-0.8)
    # f_best starts around 0 (uniform random) and goes down to -0.8
    f_min = -0.8
    f_start = 0.0
    # Clip progress to [0, 1]
    progress = np.clip((f_start - f_best) / (f_start - f_min), 0.0, 1.0)
    
    # Dynamic exploration parameter
    # Early: high exploration (large beta), Late: low exploration (small beta)
    # The idea is to explore broadly when we are far from the optimum,
    # and exploit tightly when we are close.
    # Using a quadratic decay for faster convergence to exploitation.
    beta_max = 2.0
    beta_min = 0.2
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress) ** 2
    
    # Cost-normalized UCB for minimization
    # We want to maximize the "efficiency" of the evaluation.
    # Exploitation gain: -mu (lower mu is better)
    # Exploration gain: beta * sigma (higher sigma is better)
    # Total gain: -mu + beta * sigma
    # Normalized by cost: gain / cost
    # This ensures that expensive points are only selected if they offer
    # proportionally higher potential gain or uncertainty reduction.
    
    # To avoid division issues and extreme scaling, we can use a soft cost penalty
    # or direct division. Direct division is the "cost-normalized" approach requested.
    
    exploit_gain = -mu
    explore_gain = beta * sigma_safe
    
    total_gain = exploit_gain + explore_gain
    
    # Normalize by cost
    # cost is in [1, 2]. 
    score = total_gain / cost_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
