"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lagrangian relaxation with linear score.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Calculate potential improvement (linear surrogate)
    # We are minimizing, so improvement is f_best - mu
    improvements = f_best - mu
    
    # Dynamically tune lambda
    # Use the median of positive improvements as a robust scale factor.
    # This is less sensitive to outliers than the mean.
    
    positive_imps = np.maximum(improvements, 0.0)
    pos_mask = positive_imps > 1e-12
    
    if np.any(pos_mask):
        scale = np.median(positive_imps[pos_mask])
    else:
        scale = 0.0
        
    # Use median cost for robustness, though mean and median are similar for uniform cost
    cost_scale = np.median(cost)
    
    if cost_scale > 0 and scale > 0:
        # Balance typical improvement with typical cost
        lambda_val = scale / cost_scale
    else:
        lambda_val = 0.0
        
    # Ensure lambda is non-negative
    lambda_val = max(lambda_val, 0.0)
    
    # Compute scores: f_best - mu - lambda * cost
    scores = improvements - lambda_val * cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
