"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def _elapsed():
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # GP-UCB style: linear combination of exploitation and exploration
    # Structurally different from EI: additive vs. nonlinear (CDF/PDF) form
    # score = (f_best - mu + beta * sigma) / cost
    # beta controls exploration-exploitation tradeoff
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB temperature: moderate exploration for small budget
    beta = 2.0
    
    # Lower confidence bound for minimization: we want small (mu - beta*sigma)
    # Score how much better than f_best the LCB is, per unit cost
    ucb_gain = f_best - mu + beta * sigma_safe
    
    score = ucb_gain / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
