import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 320  # 320s out of 330s budget

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are clean
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Structural Change: Quadratic Cost Penalty
    # Using cost ** 2 increases the relative weight of low-cost points,
    # which are structurally closer to the lower-bound shifted minima.
    # This sharpens the cost-awareness compared to linear division.
    beta = 2.0
    score = (-mu + beta * sigma) / (cost ** 2)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
