"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates Upper Confidence Bound (UCB) for each candidate.
    
    Strategy:
    1. Use UCB for minimization: Score = -mu + kappa * sigma.
    2. Do NOT penalize cost. Previous experiments showed that penalizing cost
       biases the search away from regions with higher x-coordinates. Since the
       global minimum's location is uniformly distributed, any bias reduces
       the probability of finding the true minimum. The budget constraint
       naturally limits expensive evaluations if they are not informative, but
       explicit penalization is counterproductive.
    3. Use a fixed kappa of 1.0, which balances exploration and exploitation
       effectively for the low budget (~20 evaluations) available.
    4. Include a deadline guard to ensure the function returns within the
       330s wall-clock limit, although the computation is very fast.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin
    
    # Ensure inputs are 1D arrays of float64 for performance and stability
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    # f_best and cost are not used in the UCB calculation directly, 
    # but we ensure they are processed if needed for future extensions.
    
    n = mu.shape[0]
    
    # Ensure sigma is positive to avoid division or log issues in other contexts,
    # though here we just multiply.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization
    # We want to minimize f. 
    # Lower mu is better (exploit).
    # Higher sigma is better (explore).
    # Score = -mu + kappa * sigma
    # Higher score means we should evaluate this point next.
    
    kappa = 1.0
    
    ucb_score = -mu + kappa * sigma_safe
    
    # Ensure finiteness: replace NaN/Inf with -inf so they are never selected
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return ucb_score
    
    return ucb_score
# EVOLVE-BLOCK-END
