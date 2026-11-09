"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 300.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Lower Confidence Bound (LCB) acquisition function.
    
    Optimizes (mu - beta * sigma) / cost, which directly targets regions with
    high risk-adjusted value density. The beta parameter balances exploration
    (high sigma) against exploitation (low mu), normalized by evaluation cost.
    """
    # Check deadline - if we're close to the limit, return a safe fallback
    if time.time() > _DEADLINE - 5.0:
        # Fallback: simple cost-normalized LCB with fixed beta
        beta = 2.0
        lcb = mu - beta * sigma
        return lcb / (cost + 1e-9)
    
    # Adaptive beta: scale with the range of mu to make it robust
    # Use a moderate exploration rate; beta=2.0 is a common choice
    beta = 2.0
    
    # Compute LCB: lower bound on the objective (we're minimizing)
    # lcb = mu - beta * sigma gives a conservative estimate of the minimum
    lcb = mu - beta * sigma
    
    # Normalize by cost to prefer cheap evaluations
    # Add small epsilon to avoid division by zero
    scores = lcb / (cost + 1e-9)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
