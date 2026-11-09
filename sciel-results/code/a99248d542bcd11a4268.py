"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 330s.
# We track the start time to ensure we return within limits, though vectorized ops are fast.
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Parameters:
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        The best observed function value so far.
    cost : np.ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns:
    scores : np.ndarray, shape (N,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # LCB for minimization:
    # We want to minimize f(x). The lower bound of the confidence interval is mu - beta * sigma.
    # We want to maximize the "potential" for improvement, which corresponds to finding
    # points where the lower bound is very low (potentially much better than f_best).
    # Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Choosing beta:
    # Standard EI uses an implicit beta that depends on the distribution.
    # A common heuristic for LCB in BO is beta ~ 1.0 to 3.0 depending on dimensionality and prior.
    # Since we are scaling coordinates to [0,1] and using Matern-5/2 with ls=0.2,
    # the uncertainty sigma is in objective units.
    # Let's start with a moderate exploration constant.
    # If beta is too high, we explore too much. If too low, we exploit too much.
    # 1.5 is a reasonable starting point for 2-3D problems.
    beta = 1.5
    
    # Calculate LCB score
    # Higher score means better (lower predicted minimum) and cheaper.
    # lcb_value = mu - beta * sigma  (lower is better for the function value)
    # acquisition_score = -lcb_value / cost = (beta * sigma - mu) / cost
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = (beta * sigma - mu) / cost
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Check deadline
    # If we are extremely close to the limit, we could return a simpler score,
    # but LCB is O(N) and very fast, so this is likely unnecessary.
    # elapsed = time.time() - _start_time
    # if elapsed > 320:
    #     return scores
    
    return scores
# EVOLVE-BLOCK-END
