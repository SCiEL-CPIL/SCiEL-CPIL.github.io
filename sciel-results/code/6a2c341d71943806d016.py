"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time across calls
_start_time = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Lower Confidence Bound (LCB) for exploitation/exploration balance,
    combined with quadratic cost penalty to maximize evaluations within budget.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    global _start_time, _DEADLINE
    
    # Check if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return simple fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-6)
        sigma = np.maximum(sigma, 1e-9)
        lcb = mu - 1.96 * sigma
        score = -lcb / (cost ** 2)
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero or negative values
    cost = np.maximum(cost, 1e-6)
    
    # Use LCB (Lower Confidence Bound) for minimization
    # LCB = mu - beta * sigma, where beta controls exploration
    # Higher beta = more exploration
    # For multimodal functions, a moderate beta works well
    beta = 1.96  # ~95% confidence level
    
    # LCB score (lower is better for minimization)
    lcb = mu - beta * sigma
    
    # Convert to maximization score (higher is better)
    lcb_score = -lcb
    
    # Cost-aware scaling: quadratic penalty to strongly prefer low-cost evaluations
    # This allows more evaluations within the fixed budget
    cost_factor = 1.0 / (cost ** 2)
    
    # Combine LCB score with cost-aware scaling
    score = lcb_score * cost_factor
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
