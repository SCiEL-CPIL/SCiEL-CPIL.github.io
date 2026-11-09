"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Cost-Modulated Lower Confidence Bound (LCB).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0

def _check_deadline():
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Cost-Modulated LCB.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Base beta for exploration
    # We want to balance exploration and exploitation.
    # Rastrigin is multimodal, so we need sufficient exploration.
    beta_base = 2.5
    
    # Modulate beta based on cost.
    # Cheaper points (lower cost) get higher beta (more exploration).
    # Expensive points (higher cost) get lower beta (more exploitation).
    # This encourages the algorithm to explore in cheap regions and exploit in expensive regions.
    # Cost is in [1, 2].
    # Let's use an inverse relationship.
    # beta(x) = beta_base * (1.5 / cost(x))
    # If cost=1, beta = 2.5 * 1.5 = 3.75
    # If cost=2, beta = 2.5 * 0.75 = 1.875
    
    beta_modulated = beta_base * (1.5 / cost_safe)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we negate it:
    # score_raw = beta * sigma - mu
    score_raw = beta_modulated * sigma - mu
    
    # Cost-aware normalization:
    # Divide by cost to get "value per cost".
    scores = score_raw / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
