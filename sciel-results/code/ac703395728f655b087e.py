"""Cost-aware acquisition for rastrigin 2d BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    Uses adaptive blending of EI and UCB with f_best-dependent weights,
    plus a small uncertainty bonus for exploration.
    """
    # Track elapsed time for safety margin
    t_start = time.time()
    deadline = t_start + 280.0  # 280s deadline with 50s margin
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound
    beta = 0.5  # exploration parameter
    ucb = mu + beta * sigma_safe
    
    # Adaptive weights based on f_best:
    # When f_best is large (early), explore more (higher UCB weight)
    # When f_best is small (late), exploit more (higher EI weight)
    # Normalize f_best relative to typical rastrigin values
    # Rastrigin on [-5.12, 5.12]^2 has values roughly 0 to ~50
    f_best_clipped = np.clip(f_best, 0.0, 50.0)
    
    # Weight for EI: increases as f_best decreases
    # Use a smooth transition
    ei_weight = 0.3 + 0.7 * np.exp(-f_best_clipped / 5.0)
    ucb_weight = 1.0 - ei_weight
    
    # Small uncertainty bonus for exploration in early stages
    unc_bonus = 0.1 * sigma_safe * np.exp(-f_best_clipped / 10.0)
    
    # Combine: weighted blend of EI and UCB, plus uncertainty bonus
    # All normalized by cost for cost-awareness
    score = (ei_weight * ei + ucb_weight * (-ucb) + unc_bonus) / np.maximum(cost, 1e-8)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
