# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Lower Confidence Bound (LCB).
    
    This approach uses a Lower Confidence Bound on the predictive mean,
    penalized by the evaluation cost. It encourages exploration of
    uncertain regions (high sigma) and exploitation of low-mean regions,
    while avoiding expensive evaluations.
    """
    # Ensure numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware penalty: divide by cost to penalize expensive points
    # Using a beta of 1.0 for a balanced exploration-exploitation trade-off
    beta = 1.0
    
    # LCB value: mu - beta * sigma
    # We want to minimize this value, so we negate it to maximize the acquisition score
    lcb_value = mu - beta * sigma_safe
    
    # Acquisition score: higher is better
    # We normalize by cost to account for evaluation expenses
    score = -lcb_value / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
