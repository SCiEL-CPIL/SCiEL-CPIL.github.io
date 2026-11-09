"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a reduced cost penalty and an exploration bonus.
    
    Args:
        mu: Array of mean predictions (2000,)
        sigma: Array of standard deviation predictions (2000,)
        f_best: Scalar, best observed value so far
        cost: Array of evaluation costs (2000,)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm.cdf and pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling
    # Using sqrt(cost) to reduce the penalty for expensive points,
    # allowing exploration of potentially better but more costly regions.
    cost_safe = np.maximum(cost, 1e-6)
    
    # Exploration bonus: favors high uncertainty regions
    # This helps escape local minima in multi-modal landscapes like Rastrigin.
    exploration_bonus = 0.5 * sigma_safe / cost_safe
    
    # Final score: EI per unit sqrt-cost + exploration bonus
    score = ei / np.sqrt(cost_safe) + exploration_bonus
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
