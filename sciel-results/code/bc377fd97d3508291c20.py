"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Uses Expected Improvement with cost-awareness and adaptive exploration.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness: normalize by cost
    # Use sqrt(cost) for softer normalization
    cost_factor = np.sqrt(cost)
    
    # Adaptive exploration: when f_best is large, explore more
    # When f_best is small, exploit more
    if f_best > 50.0:
        beta = 2.5
    elif f_best > 10.0:
        beta = 2.0
    elif f_best > 1.0:
        beta = 1.5
    else:
        beta = 1.0
    
    # UCB term
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to [0, 1] range
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-10:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Combine EI and UCB
    # Weight based on how close we are to the optimum
    if f_best < 1.0:
        ei_weight = 0.85
        ucb_weight = 0.15
    elif f_best < 10.0:
        ei_weight = 0.7
        ucb_weight = 0.3
    else:
        ei_weight = 0.5
        ucb_weight = 0.5
    
    score = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Apply cost-awareness
    score = score / cost_factor
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Return quickly if taking too long
        pass
    
    return score
# EVOLVE-BLOCK-END
