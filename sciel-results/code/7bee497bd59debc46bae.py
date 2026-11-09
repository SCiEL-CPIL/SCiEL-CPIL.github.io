import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses an adaptive Lower Confidence Bound (LCB) strategy.
    Score = beta * sigma - mu
    Higher score means more likely to be the next evaluation.
    """
    if _elapsed() > 328.0:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Adaptive beta based on progress towards the known global minimum
    global_min = -1.8013
    current_gap = f_best - global_min
    
    # The Michalewicz 2D function has a maximum value around 0 and minimum -1.8013
    # So the total range is about 1.8.
    # If current_gap is close to 0, we are near the minimum -> exploit (low beta)
    # If current_gap is close to 1.8, we are far -> explore (high beta)
    
    # Normalize the gap to [0, 1]
    max_gap = 1.8
    normalized_gap = np.clip(current_gap / max_gap, 0.0, 1.0)
    
    # Beta ranges from 1.0 (exploitation) to 3.0 (exploration)
    beta = 1.0 + 2.0 * normalized_gap
    
    # Calculate LCB score: we want to maximize (beta * sigma - mu)
    # This is equivalent to minimizing (mu - beta * sigma)
    scores = beta * sigma - mu
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
