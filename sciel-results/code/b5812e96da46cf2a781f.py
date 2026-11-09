"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we've used too much time, return a safe default
    if _get_elapsed() > 290:  # 10s safety margin before 300s limit
        # Return a reasonable default: prefer cheap points with low mu
        return -(mu + 0.1 * cost)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: when f_best is close to 0 (known minimum), 
    # reduce exploration; when f_best is large, encourage more exploration
    # f_best for rosenbrock can be quite large initially
    # Scale: when f_best > 10, explore more; when f_best < 1, exploit more
    explore_weight = np.clip(f_best / 10.0, 0.0, 1.0)
    
    # Exploration term: prefer high uncertainty regions, but weighted by cost
    # Use sigma as a proxy for uncertainty
    exploration = explore_weight * sigma / (cost + 0.1)
    
    # Cost penalty: use cost^1.5 to more strongly prefer cheap points
    # This is a structural change from linear cost penalty
    cost_penalty = cost ** 1.5
    
    # Combined score: EI per unit cost^1.5, plus exploration bonus
    # The exploration term helps avoid getting stuck in local minima
    score = ei / cost_penalty + 0.05 * exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
