import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical underflow/overflow
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive weights based on current best value
    # Griewank function range is [0, 1].
    # As f_best approaches 0, we should exploit more (trust mu).
    # As f_best is high, we should explore more (trust sigma).
    
    # Normalize f_best to [0, 1] for weight calculation
    # f_best is the lowest observed value.
    # If f_best is very small, exploit heavily.
    # If f_best is large, explore.
    
    # Clip f_best to avoid issues if it's somehow negative or > 1
    f_best_clipped = np.clip(f_best, 0.0, 1.0)
    
    # Exploitation weight: increases as f_best decreases
    # Range roughly 0.1 to 0.5
    exploit_weight = 0.1 + 0.4 * (1.0 - f_best_clipped)
    
    # Exploration weight: decreases as f_best decreases
    # Range roughly 0.3 to 0.1
    explore_weight = 0.3 * f_best_clipped + 0.1
    
    # Score components
    exploitation_term = -exploit_weight * mu
    exploration_term = explore_weight * sigma
    
    # Combine EI with adaptive UCB-like terms
    # EI is already a strong signal. The UCB terms help navigate flat regions
    # and escape local minima by balancing the pull of low mu and push of high sigma.
    
    raw_score = ei + exploitation_term + exploration_term
    
    # Cost-aware scaling
    # Dividing by cost penalizes expensive evaluations.
    # To avoid dividing by zero or very small costs (min cost is 1.0), this is safe.
    score = raw_score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
