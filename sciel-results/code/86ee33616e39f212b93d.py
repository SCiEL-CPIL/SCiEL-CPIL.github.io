"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization (griewank 2d).
    
    Uses Upper Confidence Bound (UCB) with adaptive beta that increases with
    uncertainty to encourage exploration in multimodal landscapes like Griewank.
    Cost-awareness is applied by normalizing the UCB score by evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of GP predictions at candidate points
    sigma : array-like, shape (2000,)
        Standard deviation of GP predictions at candidate points
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (in [1, 2])
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB: beta increases with overall uncertainty to encourage exploration
    # In early stages (high sigma), we want more exploration; later, more exploitation
    mean_sigma = np.mean(sigma)
    
    # Beta scales with uncertainty: higher uncertainty -> more exploration
    # Use a logarithmic scaling to keep beta in a reasonable range
    beta = 1.0 + 2.0 * np.log1p(mean_sigma / (abs(f_best) + 1e-6))
    
    # UCB score: we want to minimize, so we use -mu + beta * sigma
    # Higher UCB score = more promising (lower predicted value + more uncertainty)
    ucb_score = -mu + beta * sigma
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = ucb_score / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

# EVOLVE-BLOCK-END
