"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses UCB per unit cost with adaptive beta based on exploration progress.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions in objective units
    sigma: array of shape (2000,) - GP standard deviations in objective units
    f_best: float - lowest observed value so far
    cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute UCB: mu + beta * sigma
    # Adaptive beta: start higher for exploration, decrease as we get closer to optimum
    # Use a simple heuristic: beta decreases as f_best approaches 0 (the known minimum)
    # This encourages more exploration early and exploitation later
    
    # Estimate progress: how much have we improved from initial random points?
    # Since we don't have initial values, use a relative measure
    # If f_best is close to 0, we're doing well -> reduce beta
    # If f_best is still large, keep beta high
    
    # Scale f_best to get a progress measure (functions have min 0 or -3.86)
    # Use a simple decay: beta starts at 3.0 and decays
    # The cost sum gives us a rough idea of how many evaluations we've done
    
    # Use the mean cost as a proxy for number of evaluations (since cost ~ 1-2)
    mean_cost = np.mean(cost)
    
    # Start with high exploration, reduce over time
    # beta = base_beta / (1 + progress_factor)
    # progress_factor based on how close f_best is to the known minimum
    # For functions with min 0: progress = 1 / (1 + |f_best|)
    # For hartmann with min -3.86: need to normalize
    
    # Simple approach: use a fixed beta that balances well
    # Based on the observation that UCB with beta=2 improved from 1.0 to 1.219
    # Let's try an adaptive approach
    
    # Estimate scale of the objective function
    # The spread of mu values gives us a sense of the function scale
    mu_range = np.ptp(mu) if len(mu) > 0 else 1.0
    
    # Adaptive beta: higher when uncertainty is high relative to improvement
    # beta = 2.0 + 1.0 * (sigma_mean / (|f_best| + mu_range + 1e-6))
    sigma_mean = np.mean(sigma)
    
    # This gives beta in range [2.0, 3.0] typically
    beta = 2.0 + 1.0 * (sigma_mean / (abs(f_best) + mu_range + 1e-6))
    beta = np.clip(beta, 1.5, 4.0)
    
    # UCB score: mu + beta * sigma (we want to minimize, so higher is better for acquisition)
    ucb = mu + beta * sigma
    
    # Cost-aware: divide by cost
    scores = ucb / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
