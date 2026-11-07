"""Cost-aware acquisition using GP-LCB with adaptive exploration and cost normalization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Module-level persistent state across calls within one evaluation run
_STATE = {
    "start_time": None,
    "f_best_history": [],
    "call_count": 0
}


def acquisition(mu, sigma, f_best, cost):
    """
    GP-LCB Cost-Aware Acquisition Function.
    
    Uses Lower Confidence Bound with adaptive exploration parameter,
    normalized by evaluation cost. The exploration parameter adapts
    based on recent improvement history to balance exploration and exploitation.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if _STATE["start_time"] is None:
        _STATE["start_time"] = time.time()
    
    elapsed = time.time() - _STATE["start_time"]
    if elapsed > 290.0:  # Safety margin
        # Fallback to simple cost-aware LCB
        sigma_safe = np.maximum(sigma, 1e-10)
        lcb = f_best - mu + 1.5 * sigma_safe
        scores = lcb / np.maximum(cost, 1e-10)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Update state
    _STATE["call_count"] += 1
    _STATE["f_best_history"].append(f_best)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma close to 0
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Adaptive exploration parameter (beta)
    # Start with moderate exploration, reduce if stagnating
    beta = 1.5
    if len(_STATE["f_best_history"]) >= 3:
        # Check improvement over last 2 steps
        recent_improvement = _STATE["f_best_history"][-3] - _STATE["f_best_history"][-1]
        if recent_improvement > 0.01:
            # Significant improvement, keep exploring
            beta = 1.5
        elif recent_improvement > 0.001:
            # Moderate improvement, balance
            beta = 1.0
        else:
            # Stagnation, exploit more
            beta = 0.5
            
    # GP-LCB: minimize mu - beta * sigma
    # Score = -(mu - beta * sigma) / cost = (f_best - mu + beta * sigma) / cost
    # We add f_best as a baseline to make scores more interpretable
    lcb_score = f_best - mu + beta * sigma_safe
    
    # Normalize by cost
    scores = lcb_score / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
