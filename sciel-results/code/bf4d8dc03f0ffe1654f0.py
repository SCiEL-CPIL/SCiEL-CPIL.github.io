"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed function value
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a valid finite array to avoid invalidating the result
        # If we run out of time, we return a random but finite score
        return np.random.uniform(size=mu.shape)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB for minimization: we want to minimize (mu - beta * sigma)
    # Since we are maximizing the acquisition score, we return negative LCB
    # Or equivalently, we can view it as maximizing ( -mu + beta * sigma )
    
    # Tuning parameter beta controls exploration vs exploitation.
    # For Rosenbrock, a moderate exploration is needed to find the valley.
    # Standard LCB uses beta ~ 2-5. We use a dynamic or fixed beta.
    # Let's try a fixed beta of 2.0 which is a common robust choice.
    beta = 2.0
    
    # Calculate LCB score: -mu + beta * sigma
    lcb_scores = -mu + beta * sigma_safe
    
    # Normalize by cost to account for cost-awareness
    # Cost is in [1, 2] roughly (1 + mean(x in [0,1])).
    # Dividing by cost penalizes expensive evaluations.
    cost_safe = np.maximum(cost, 1e-8)
    scores = lcb_scores / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
