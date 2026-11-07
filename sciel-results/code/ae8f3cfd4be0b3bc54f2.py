import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition for Styblinski-Tang 2D BO.
    Uses Lower Confidence Bound divided by cost to target high-value, low-cost regions.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Progress measure based on f_best
    known_min = -78.3323
    progress = np.clip(1.0 - (f_best - known_min) / 200.0, 0.0, 1.0)
    
    # Adaptive beta for LCB: start high (explore), decrease over time
    # LCB = mu - beta * sigma, we want to MINIMIZE this
    # So acquisition = -LCB / cost = (beta * sigma - mu) / cost
    beta = 4.0 - 3.0 * progress
    
    # LCB: lower confidence bound (we want to minimize this)
    lcb = mu - beta * sigma
    
    # Acquisition: negative LCB divided by cost
    # Higher score = better candidate
    score = -lcb / (cost ** 1.0)
    
    # Add a small exploration bonus for unexplored regions (high sigma)
    # This helps escape local optima early in the search
    exploration_bonus = 0.5 * (1.0 - progress) * sigma / (sigma + 1.0)
    score += exploration_bonus / (cost ** 1.0)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Final safety check on time
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    return score
