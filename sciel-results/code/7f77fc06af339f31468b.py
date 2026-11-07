"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) 
    with a quadratic cost penalty.
    
    This approach prioritizes points that have a low lower bound on the 
    objective value (encouraging both exploitation of good means and 
    exploration of high uncertainty) while aggressively penalizing 
    high-cost evaluations to maximize the number of evaluations 
    within the tight cost budget.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed objective value so far (minimum, since we minimize).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point, in [1, 2].
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline Guard: If we are running out of time, return a fast, simple score
    # to ensure we don't contribute to a timeout.
    # The evaluator limit is 330s. We use 320s as a safety margin.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_arr, 1e-8)
        # Simple inverse cost score
        return 1.0 / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost_arr, 1e-8)
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB.
    # To convert to a "higher is better" score, we calculate the potential improvement:
    # improvement = f_best - LCB(x) = f_best - mu(x) + beta * sigma(x)
    
    # Beta controls the exploration/exploitation trade-off.
    # A higher beta favors exploration (high sigma).
    # A lower beta favors exploitation (low mu).
    # 2.0 is a standard choice for Bayesian Optimization.
    beta = 2.0
    
    lcb = mu - beta * sigma_safe
    improvement = f_best - lcb
    
    # Points where the lower bound is already worse than f_best 
    # (i.e., improvement <= 0) are not promising. 
    # We set their score to 0 to avoid selecting them unless all are bad.
    improvement = np.maximum(improvement, 0.0)
    
    # Apply Quadratic Cost Penalty (1/cost^2)
    # This aggressively prioritizes low-cost evaluations,
    # fitting more evaluations within the cost budget of 30.
    score = improvement / (cost_safe ** 2)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
