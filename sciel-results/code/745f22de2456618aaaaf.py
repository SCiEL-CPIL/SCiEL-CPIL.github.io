# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (griewank 2d).
    
    Uses a hybrid approach:
    1. Expected Improvement per unit cost for exploitation
    2. Thompson Sampling-inspired exploration using sigma
    3. Adaptive weighting based on how close we are to the known minimum
    
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
    
    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI
    ei_cost = ei / cost
    
    # Exploration component: use sigma as a proxy for uncertainty
    # Scale by 1/cost to be cost-aware
    exploration = sigma / cost
    
    # Adaptive weight: more exploration when f_best is far from 0 (the true minimum)
    # For griewank 2d, the minimum is 0, so if f_best is large, we need more exploration
    # If f_best is close to 0, we can exploit more
    # Use a smooth transition
    f_best_abs = abs(f_best)
    # When f_best is large (>0.5), weight exploration more; when small (<0.01), weight exploitation more
    # Use a sigmoid-like function
    exploration_weight = 0.5 / (1.0 + np.exp(-5.0 * (f_best_abs - 0.1)))
    
    # Combine exploitation and exploration
    scores = (1.0 - exploration_weight) * ei_cost + exploration_weight * exploration
    
    # Add a small bonus for points that are predicted to be very low (near zero)
    # This helps find the global minimum of griewank
    low_pred_bonus = np.exp(-mu / max(f_best, 1e-6)) / cost
    scores += 0.05 * low_pred_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
