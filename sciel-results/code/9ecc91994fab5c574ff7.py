"""Cost-aware acquisition using LCB with adaptive beta for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) with
    adaptive exploration parameter for Griewank 2D Bayesian Optimization.
    
    LCB uses a linear combination of the mean and standard deviation,
    which is structurally different from EI or PoI. The beta parameter
    is adapted based on the relative uncertainty to balance exploration
    and exploitation.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    deadline = t0 + 290.0  # safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute LCB: for minimization, we want low mu and high sigma
    # LCB = mu - beta * sigma (lower is better for the objective)
    # We maximize -LCB = -mu + beta * sigma
    # 
    # Adaptive beta: use a moderate exploration weight
    # Scale beta based on the spread of mu values to keep the balance
    mu_range = np.ptp(mu) if n > 1 else 1.0
    sigma_range = np.ptp(sigma_safe) if n > 1 else 1.0
    
    # Base beta that balances exploitation and exploration
    # Use a value that makes the sigma term comparable to mu spread
    if mu_range > 1e-12:
        beta_base = 2.0 * mu_range / (sigma_range + 1e-12)
    else:
        beta_base = 2.0
    
    # Clamp beta to reasonable range
    beta = np.clip(beta_base, 0.5, 10.0)
    
    # LCB score: higher is better (we want to minimize, so negative LCB)
    # score = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-aware penalty: penalize expensive evaluations
    # Use a multiplicative penalty that's gentle for low cost, stronger for high cost
    # Cost is in [1, 2] range (1 + mean(x_scaled) where x_scaled in [0,1])
    cost_penalty = 1.0 / np.sqrt(cost_safe)
    
    # Combine
    score = lcb_score * cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() > deadline:
        # Return a safe fallback: just the negative mean weighted by cost
        fallback = -mu * cost_penalty
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
# EVOLVE-BLOCK-END
