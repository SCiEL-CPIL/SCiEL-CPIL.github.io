import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Safety check for timeout (though this function is fast)
    # If we are close to the limit, return a simple heuristic
    if time.time() - _start_time > 250:
         return -cost  # Favor cheap points as a last resort

    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Numerical stability
    sigma = np.clip(sigma, 1e-8, 1e8)
    cost = np.clip(cost, 1e-6, 10.0) # Cost is [1,2] theoretically, but safe bounds
    
    # Calculate Expected Improvement (EI) for minimization
    # delta = f_best - mu. If mu < f_best, delta > 0 (potential improvement)
    delta = f_best - mu
    
    # z-score
    z = delta / sigma
    
    # Use scipy for accurate CDF/PDF
    # Clip z to avoid numerical issues in extreme tails if necessary, 
    # though scipy is generally robust.
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = delta * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    
    # Structural Change 1: Logarithmic Cost-Aware Normalization
    # Unlike 'ei - k*cost' or 'ei / cost', this uses a damping factor.
    # As cost increases, the effective EI decreases non-linearly.
    # beta controls the strength of cost aversion.
    # Cost is in [1, 2]. 
    # If cost=1, denom = 1 + 0.5*1 = 1.5
    # If cost=2, denom = 1 + 0.5*2 = 2.0
    # Ratio of effective EI at cost 1 vs 2 is 2.0/1.5 = 1.33x.
    # This is a moderate penalty, distinct from the 2x penalty of division.
    beta = 0.5
    cost_factor = 1.0 + beta * cost
    ei_cost_aware = ei / cost_factor
    
    # Structural Change 2: Adaptive Exploration Bonus
    # The current program uses a fixed 0.1 * sigma.
    # Here, we scale exploration by the "urgency" of the search.
    # If f_best is high (early in search), we explore more.
    # As f_best -> 0, exploration dampens.
    # f_best is the lowest observed value. 
    # Three-hump camel min is 0.
    # Scale: 1.0 + f_best ensures we never divide by zero and keeps weight >= 1.0 initially.
    # gamma = 0.05
    # Exploration term: gamma * sigma * (1.0 + f_best)
    # This is structurally different from a constant multiplier on sigma.
    gamma = 0.05
    exploration = gamma * sigma * (1.0 + np.abs(f_best))
    
    # Combine
    score = ei_cost_aware + exploration
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
