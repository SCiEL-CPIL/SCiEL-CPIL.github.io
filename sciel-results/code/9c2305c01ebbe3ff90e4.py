"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) 
    for minimization.
    
    This function computes a score based on the Lower Confidence Bound 
    (optimistic prediction) of the objective function, divided by the 
    evaluation cost. This encourages exploring regions with high uncertainty 
    (high sigma) while penalizing regions with high predicted cost.
    
    The score is defined as:
    score = (beta * sigma - (mu - f_best)) / cost
    
    Where:
    - mu: GP mean prediction
    - sigma: GP standard deviation prediction
    - f_best: Best observed objective value
    - cost: Cost of evaluating the point
    - beta: Exploration parameter (set to 2.0)
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Ensure inputs are float64 arrays for numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Exploration parameter for LCB
    # 2.0 is a common heuristic for 2D problems
    beta = 2.0
    
    # Calculate LCB score
    # We want to maximize the potential for improvement.
    # The optimistic value is mu - beta * sigma.
    # Improvement over f_best is f_best - (mu - beta * sigma) = f_best - mu + beta * sigma.
    # Score = Improvement / Cost
    score = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Ensure scores are non-negative
    # Points with predicted value much higher than f_best (even with uncertainty) get 0
    score = np.maximum(score, 0.0)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: return immediately if we are approaching the limit
    elapsed = time.time() - t0
    if elapsed > 290.0:
        return score
        
    return score
# EVOLVE-BLOCK-END
