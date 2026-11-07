"""Rewritten acquisition function using Cost-Aware UCB with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Structural change from EI/cost to UCB/cost with adaptive beta.
    UCB explicitly balances exploitation (low mu) and exploration (high sigma).
    The exploration coefficient beta is adapted based on f_best:
    If f_best is far from the known minimum (0), we explore more (higher beta).
    If f_best is close to 0, we exploit more (lower beta).
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: simple cost-aware random exploration or uniform
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_arr = np.where(np.isfinite(cost_arr) & (cost_arr > 0), cost_arr, 1.0)
        # In case of timeout, we just return 1/cost to pick cheap points
        return 1.0 / cost_arr

    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Ensure f_best is a scalar float
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
        
    # Adaptive Beta:
    # Base exploration is 2.0.
    # If f_best is large (bad), we increase exploration to find better regions.
    # If f_best is small (good), we decrease exploration to refine.
    # Using a linear ramp: beta = 2.0 + f_best_val
    # Capping beta to avoid extreme behavior if f_best is unexpectedly huge
    beta = 2.0 + np.clip(f_best_val, 0.0, 10.0)
    
    # UCB for minimization:
    # We want to minimize y.
    # UCB lower bound: L(x) = mu(x) - beta * sigma(x)
    # We want to select x that minimizes L(x) / cost(x) ?
    # No, we want to maximize the "potential improvement per cost".
    # The "value" of a point is how low it could be.
    # Potential low value = mu - beta * sigma.
    # We want this to be as small as possible.
    # So we want to MINIMIZE (mu - beta * sigma) / cost.
    # Equivalently, we want to MAXIMIZE (beta * sigma - mu) / cost.
    
    numerator = beta * sigma - mu
    scores = numerator / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
