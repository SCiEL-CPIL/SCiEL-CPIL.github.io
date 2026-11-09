# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Strategy: Cost-Adjusted Upper Confidence Bound (UCB) with a fixed 
    exploration coefficient (beta=2.5). This structural change removes 
    the heuristic adaptive weights and the ineffective Probability of 
    Improvement term from the previous hybrid approach, focusing instead 
    on a robust balance between exploitation (low mean) and exploration 
    (high uncertainty) that is better suited for the multimodal Rastrigin 
    function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior in objective units.
    sigma : ndarray, shape (2000,)
        Standard deviation of GP posterior in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        mu = np.where(np.isfinite(mu), mu, f_best)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Cost-Adjusted UCB for minimization ---
    # Score = (-mu + beta * sigma) / cost
    # We want to maximize this score.
    # -mu rewards low means (exploitation).
    # beta * sigma rewards high uncertainty (exploration).
    # Dividing by cost penalizes expensive evaluations.
    
    # Fixed beta for robust exploration in multimodal landscapes
    beta = 2.5
    
    ucb_score = -mu + beta * sigma_safe
    
    # Cost normalization
    scores = ucb_score / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
