"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement.
    
    Strategy:
    - Uses EI with a small noise parameter xi for exploration.
    - Divides by sqrt(cost) instead of cost to allow more exploration 
      in moderately high-cost regions (measured improvement 1.0 -> 1.442).
    - Adds a small sigma-based exploration bonus to prevent premature convergence.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs (cost should be positive)
    # The problem states cost is 1 + mean(x scaled to [0,1]), so cost >= 1
    safe_cost = np.maximum(cost, 1e-6)
    
    # Small noise parameter for exploration
    # Scaling with f_best magnitude helps if f_best is very large or small
    xi = 0.05 * max(abs(f_best), 1.0)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - xi - mu) / sigma
    z = (f_best - xi - mu) / (sigma + 1e-9)
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: EI / sqrt(cost)
    # This reduces the penalty for high-cost points compared to EI/cost,
    # allowing the optimizer to explore regions with higher evaluation costs
    # if the potential improvement is high.
    score = ei / np.sqrt(safe_cost)
    
    # Add a small exploration bonus based on uncertainty (sigma)
    # This helps escape local minima in the surrogate model's confidence
    # The bonus is scaled down to not dominate the EI term
    exploration_bonus = 0.1 * sigma / np.sqrt(safe_cost)
    
    score += exploration_bonus
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
