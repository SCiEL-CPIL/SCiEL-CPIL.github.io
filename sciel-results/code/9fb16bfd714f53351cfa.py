"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rastrigin 2D.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit
    
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Clip cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score: z = (f_best - mu) / sigma
    # Positive z means mu < f_best (potential improvement)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement with small xi for slightly aggressive exploitation
    xi = 0.01
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost to prefer cheaper evaluations
    ei_per_cost = ei / cost_safe
    
    # Exploration bonus: reward high uncertainty regions
    # This helps escape local minima in rastrigin
    # Scale exploration weight relative to typical EI magnitude
    # Use a moderate exploration weight that balances exploitation and exploration
    exploration_weight = 0.15
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Combine exploitation and exploration
    scores = ei_per_cost + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores
    scores = np.maximum(scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
