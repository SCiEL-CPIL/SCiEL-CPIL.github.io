"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Lower Confidence Bound (LCB).
    
    Args:
        mu: np.array of shape (N,) predicted means.
        sigma: np.array of shape (N,) predicted standard deviations.
        f_best: float, the best observed value so far.
        cost: np.array of shape (N,) evaluation costs.
        
    Returns:
        np.array of shape (N,) acquisition scores (higher is better).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a neutral score if we are over budget, though this shouldn't happen
        # for vectorized operations on 2000 points.
        return np.zeros_like(mu)
    
    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost (though cost >= 1 by definition)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Lower Confidence Bound for minimization:
    # We want to minimize f(x). The LCB is mu - beta * sigma.
    # A point is attractive if its LCB is low.
    # Score should be high for attractive points.
    # Score = (f_best - LCB) / cost = (f_best - (mu - beta * sigma)) / cost
    #       = (f_best - mu + beta * sigma) / cost
    
    # Tuning parameter beta for exploration
    beta = 1.0
    
    lcb = mu - beta * sigma
    # Regret potential relative to current best
    regret_potential = f_best - lcb
    
    # Cost-aware score
    scores = regret_potential / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
