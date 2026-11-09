import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive parameter tuning
_state = {
    'best_f': np.inf,
    'f_history': [],
    'n_calls': 0,
    'start_time': None
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Lower Confidence Bound (LCB/UCB)
    for minimization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = prefer)
    """
    global _state
    
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['n_calls'] += 1
    _state['f_history'].append(f_best)
    
    elapsed = time.time() - _state['start_time']
    
    # Safety: if we're close to timeout, use a fast fallback
    if elapsed > 280:
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        cost_safe = np.where(cost > 1e-9, cost, 1e-9)
        # Fast UCB fallback
        beta = 1.0
        lcb = mu - beta * sigma_safe
        scores = -lcb / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Track progress
    if f_best < _state['best_f']:
        _state['best_f'] = f_best
    
    n_calls = _state['n_calls']
    
    # Adaptive Beta for UCB
    # Budget is ~30 cost. Avg cost ~1.5. Max ~20 evals.
    # We start with 5 evals. So we have ~15 evals for the loop.
    # Let's assume total evals will be around 20-25.
    
    # Progress from 0 to 1 over 20 calls
    # n_calls starts at 1 (first call to acquisition is after 5 initial points? 
    # No, "Runs start with 5 uniform random points". Then "evaluator scores 2000 new... and evaluates the best one".
    # So the first call to acquisition is for the 6th point.
    # n_calls in _state is incremented at the start of the function.
    # So first call: n_calls = 1.
    # Last call: n_calls ~ 15-20.
    
    # Normalize progress
    # If n_calls is 1, progress is 0.
    # If n_calls is 20, progress is 1.
    max_expected_calls = 20
    progress = min(1.0, max(0.0, (n_calls - 1) / max_expected_calls))
    
    # Beta decay:
    # Start high (exploration), end low (exploitation)
    # Quadratic decay for smoother transition
    beta_max = 2.5
    beta_min = 0.5
    beta = beta_min + (beta_max - beta_min) * (1 - progress) ** 2
    
    # Compute LCB (Lower Confidence Bound) for minimization
    lcb = mu - beta * sigma_safe
    
    # Cost-aware scoring
    scores = -lcb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
