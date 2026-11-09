"""
Cost-aware acquisition function for Bayesian optimization (powell 4d).
Combines the structural approach of the seed (myopic acquisition) with the 
adaptive LCB strategy and state management from the highest-scoring inspiration.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for adaptive parameter tuning and safety
_state = {
    'n_calls': 0,
    'start_time': None
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Lower Confidence Bound (LCB)
    for minimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _state
    
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['n_calls'] += 1
    
    elapsed = time.time() - _state['start_time']
    
    # Safety: if we're close to timeout, use a fast, simple fallback
    # The budget is ~30 cost, implying ~20-25 evaluations.
    # Each call should be fast.
    if elapsed > 280:
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        # Simple UCB/LCB fallback with fixed beta
        beta = 1.0
        lcb = mu - beta * sigma_safe
        scores = -lcb / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: sigma should be positive, cost should be positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    n_calls = _state['n_calls']
    
    # Adaptive Beta for LCB
    # We estimate the total number of calls based on the cost budget.
    # Budget = 30. Initial 5 points (cost ~7.5). Remaining ~22.5.
    # Avg cost per step ~1.5. Expected remaining steps ~15.
    # Total steps ~20.
    max_expected_calls = 20
    progress = min(1.0, max(0.0, (n_calls - 1) / max_expected_calls))
    
    # Beta decay:
    # Start with high exploration (beta_max), end with exploitation (beta_min).
    # Quadratic decay allows for a longer exploration phase initially.
    beta_max = 2.5
    beta_min = 0.5
    beta = beta_min + (beta_max - beta_min) * (1 - progress) ** 2
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # We want to minimize f, so we look for low mu and high sigma.
    # LCB = mu - beta * sigma
    # We maximize the score, so we use -LCB = beta * sigma - mu
    lcb = mu - beta * sigma_safe
    scores = -lcb / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
