import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = None


def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Hybrid UCB with adaptive exploration and cost normalization.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed function value
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 300s limit, return a simple fallback
    if _get_elapsed_time() > 290:
        mu = np.asarray(mu)
        cost = np.asarray(cost)
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute the gap between f_best and mu
    gap = f_best - mu
    
    # Use a UCB-style acquisition with adaptive beta
    # The idea: use a larger exploration bonus when sigma is high relative to the gap
    # This helps escape local optima on multimodal landscapes like Griewank
    
    # Adaptive beta: larger when uncertainty is high relative to improvement potential
    # beta scales with the ratio of sigma to the gap (when gap > 0)
    # When gap <= 0 (mu < f_best), we're already better than best, so pure exploitation
    positive_gap = np.maximum(gap, 0.0)
    
    # Compute a normalized uncertainty measure
    # When gap is small, uncertainty matters more -> higher beta
    # When gap is large, exploitation matters more -> lower beta
    # Use a soft transition: beta = base_beta * (1 + uncertainty_factor)
    
    # uncertainty_factor: high when sigma is large relative to gap
    # Use a robust measure: sigma / (positive_gap + epsilon)
    eps = 1e-6
    uncertainty_ratio = sigma_safe / (positive_gap + eps)
    
    # Clip the ratio to avoid extreme values
    uncertainty_ratio = np.clip(uncertainty_ratio, 0.0, 10.0)
    
    # Adaptive beta: starts at 0.8 for pure exploitation, increases with uncertainty
    # The +1.0 term ensures a minimum exploration bonus
    beta = 0.8 + 0.4 * np.tanh(uncertainty_ratio / 2.0)
    
    # UCB score: -mu + beta * sigma
    # Higher is better (we want to minimize, so -mu rewards low values)
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware: score per unit cost
    # Normalize by cost to prefer cheaper evaluations
    score = ucb / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
