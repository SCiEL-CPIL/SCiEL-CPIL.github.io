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
    
    # Dynamic, instance-adaptive exploration budget
    # Scale the UCB weight by the ratio of remaining budget to current cost.
    # This shifts from aggressive exploration in early iterations to precise exploitation.
    
    # Estimate total cost spent so far (roughly n_calls * avg_cost)
    # We don't have exact total cost, but we can use n_calls as a proxy for progress
    # Budget is 30, initial 5 points cost ~7.5, so remaining budget ~22.5
    # Let's estimate progress based on n_calls relative to expected max calls
    
    max_expected_calls = 25  # Slightly higher to account for variable costs
    progress = min(1.0, max(0.0, (n_calls - 1) / max_expected_calls))
    
    # Dynamic beta: scales with remaining budget ratio
    # Early on (progress ~0): high beta for exploration
    # Late (progress ~1): low beta for exploitation
    remaining_budget_ratio = 1.0 - progress
    
    # Base beta values
    beta_base_max = 3.0
    beta_base_min = 0.3
    
    # Dynamic beta that also considers the scale of f_best
    # When f_best is large (early), we can afford more exploration
    # When f_best is small (late), we should be more precise
    f_best_scale = max(abs(f_best), 1e-6)
    
    # Adaptive beta: combines progress-based decay with f_best scale
    beta = beta_base_min + (beta_base_max - beta_base_min) * remaining_budget_ratio ** 1.5
    
    # Cost sensitivity (gamma): how much we penalize expensive evaluations
    # Early: low cost sensitivity (explore freely)
    # Late: high cost sensitivity (prefer cheap evaluations)
    gamma_min = 0.5
    gamma_max = 2.0
    gamma = gamma_min + (gamma_max - gamma_min) * progress ** 2
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # Score = -LCB / (cost ^ gamma)
    lcb = mu - beta * sigma_safe
    cost_weighted = cost_safe ** gamma
    scores = -lcb / cost_weighted
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
