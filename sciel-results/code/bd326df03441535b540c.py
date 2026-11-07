import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    global _START_TIME
    if _START_TIME is None:
        return float('inf')
    # 330s hard limit, use 300s as safety margin
    return 300.0 - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Probability of Improvement (POI) for robust exploitation 
    with a UCB-style term for exploration, both normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    _init_time()
    
    # Deadline guard: if we are close to the limit, return a simple, fast score
    if _time_remaining() < 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        scores = poi / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Standard robust calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Probability of Improvement (POI)
    # POI is P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is a robust exploitation metric, especially in low-data regimes.
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # 2. UCB-style Exploration Term
    # We want to explore regions with high uncertainty (high sigma) 
    # and potentially lower mean.
    # UCB for minimization: f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    # We clamp to 0 to ensure we don't reward points that are already predicted 
    # to be worse than f_best by a large margin without uncertainty.
    beta = 2.0
    ucb_raw = f_best - mu + beta * sigma_safe
    
    # Scale the UCB term to be comparable to POI (which is in [0, 1])
    # Use the magnitude of f_best as a natural scale, with a floor to avoid division by small numbers
    scale = np.abs(f_best) + 1.0
    ucb_scaled = np.maximum(ucb_raw, 0.0) / scale
    
    # 3. Combine POI and UCB
    # Weighting: POI is primary (exploitation), UCB is secondary (exploration)
    # In Inspiration 1 (POI only), score was 1.22. 
    # In Inspiration 2 (EI + 0.3*UCB), score was 1.19.
    # We try a blend: 1.0 * POI + 0.5 * UCB_scaled
    # The 0.5 weight encourages exploration without overwhelming the POI signal.
    combined_score = poi + 0.5 * ucb_scaled
    
    # 4. Cost Normalization
    # Divide by cost to prefer cheaper evaluations
    scores = combined_score / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
