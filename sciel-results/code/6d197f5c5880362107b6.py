import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a cost-adjusted Lower Confidence Bound (LCB) formulation where:
    - The exploration term (beta * sigma) is scaled by 1/cost, so high-cost regions
      get a smaller uncertainty bonus.
    - The mean term remains invariant to cost.
    - This creates a non-uniform exploration landscape that prioritizes 
      high-uncertainty, low-cost areas.
    
    Additionally, we apply a mild overall cost penalty to bias toward cheaper evaluations
    while still allowing exploration of the valid minimum region (inner 80% of domain).
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware LCB:
    # Standard LCB for minimization: score = mu - beta * sigma
    # We want higher scores to be better, so we negate: score = -(mu - beta * sigma) = -mu + beta * sigma
    # But we scale the exploration by 1/cost to penalize high-cost regions in uncertainty.
    # 
    # Adaptive beta: we use a fixed beta that balances exploration/exploitation.
    # For short budgets (~30 cost units), we want moderate exploration.
    
    # Base exploration coefficient
    beta = 1.5
    
    # Scale exploration by inverse cost: high cost -> less exploration bonus
    # cost is in [1.0, 2.0] typically (1 + mean(x_scaled))
    # 1/cost ranges from 0.5 to 1.0
    cost_inverse = 1.0 / np.maximum(cost, 1e-9)
    
    # LCB score (higher is better)
    # We use: score = -mu + beta * sigma * (1/cost)
    # The -mu term favors low predicted values (exploitation)
    # The beta * sigma / cost term favors high uncertainty in low-cost regions (exploration)
    lcb_score = -mu + beta * sigma_safe * cost_inverse
    
    # Additional mild cost penalty to further bias toward cheaper points
    # Use a gentle exponential decay centered at cost=1.2 (middle of valid range)
    # This is mild so as not to exclude the valid minimum region
    cost_penalty = np.exp(-0.3 * (cost - 1.2) ** 2)
    
    # Combine LCB score with cost penalty
    scores = lcb_score * cost_penalty
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
