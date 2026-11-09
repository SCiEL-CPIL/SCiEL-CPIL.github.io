"""Cost-aware acquisition for Ackley 2D BO with deadline guard and PI-based scoring."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        return np.nan_to_num(-mu_safe / cost_safe, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    # Structural Change: Switch from Expected Improvement (EI) to UCB (Upper Confidence Bound)
    # UCB for minimization: Score = -mu + beta * sigma
    # This promotes exploration in high-uncertainty regions, which is crucial for multi-modal
    # functions like Ackley when the budget is tight and initial data is sparse.
    
    beta = 1.5  # Exploration-exploitation trade-off parameter
    
    # Calculate UCB score
    ucb = -mu + beta * sigma
    
    # Cost-aware scoring: Divide by cost to prefer cheaper evaluations
    scores = ucb / cost
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
