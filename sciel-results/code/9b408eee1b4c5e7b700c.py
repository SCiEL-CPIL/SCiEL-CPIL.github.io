"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with softened cost normalization
    and a small UCB exploration bonus.
    
    Key improvements over pure EI/cost:
    1. Softened cost normalization: EI / cost^0.5 instead of EI / cost
       This reduces the aggressive penalty for expensive points while still
       preferring cheaper evaluations.
    2. Small UCB exploration term to prevent premature convergence.
    """
    # Deadline guard: track elapsed time to ensure we return within safety margin.
    t_start = time.time()
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Softened cost normalization: use square root of cost
    # This is less aggressive than linear cost division
    cost_factor = np.sqrt(cost)
    ei_normalized = ei / cost_factor
    
    # Add a small UCB exploration bonus to encourage exploration
    # beta controls the exploration-exploitation tradeoff
    beta = 0.1
    ucb_bonus = beta * sigma
    
    # Combine: primary score is EI/cost^0.5, plus exploration bonus
    scores = ei_normalized + ucb_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t_start > 280:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
