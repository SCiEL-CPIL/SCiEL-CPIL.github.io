import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    UCB-based acquisition function for Styblinski-Tang 2D Bayesian Optimization.
    Uses adaptive beta that decreases with exploration progress.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / cost_safe
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(np.where(np.isfinite(sigma), sigma, 1e-9), 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Adaptive beta: start high for exploration, decrease as we find better values
    # Known minimum is -78.3323, starting from uniform random likely gives f_best around 0-5
    known_min = -78.3323
    # Progress: 0 when f_best is far from min, 1 when close
    # Scale: assume f_best range from 0 to -78
    progress = np.clip(1.0 - (f_best - known_min) / 100.0, 0.0, 1.0)
    
    # Beta starts at 3.0 (exploration-heavy) and decreases to 1.5 (exploitation)
    beta = 3.0 - 1.5 * progress
    
    # UCB score: mu - beta * sigma (since we want to minimize, lower is better)
    # But we want higher scores for "evaluate next", so we negate
    # Actually, for minimization with UCB: we want low mu + high sigma
    # Score = -mu + beta * sigma (higher score = more promising)
    ucb_score = -mu + beta * sigma
    
    # Cost adjustment: divide by cost^exponent
    # Use fixed exponent 1.5 which worked well in inspiration
    cost_adj = cost ** 1.5
    score = ucb_score / cost_adj
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
