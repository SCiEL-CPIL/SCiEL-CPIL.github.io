"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Combines:
    - Expected Improvement (EI) for exploitation
    - Upper Confidence Bound (UCB) for exploration
    - Adaptive exploration parameter (beta) based on uncertainty and f_best
    - Strong cost-awareness via (1 + cost) normalization
    - Explicit uncertainty bonus to escape local minima
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
    
    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute UCB-style exploration term
    # Adaptive beta: 
    # - Base value
    # - Boost if overall uncertainty is high (early GP stages)
    # - Boost if f_best is far from optimum (multimodal landscape)
    mean_sigma = np.mean(sigma)
    beta = 1.5
    
    # Uncertainty boost: decays as GP becomes confident
    uncertainty_boost = 2.0 * np.exp(-mean_sigma * 5.0)
    beta += uncertainty_boost
    
    # f_best boost: encourage more exploration when far from the minimum
    if f_best > 50.0:
        beta += 1.5
    elif f_best > 10.0:
        beta += 0.8
    elif f_best > 1.0:
        beta += 0.2
        
    ucb = -mu + beta * sigma  # Lower mu is better, so we want high -mu + beta*sigma
    
    # 3. Normalize EI and UCB to comparable [0, 1] scales
    ei_max = np.max(ei)
    if ei_max < 1e-10:
        ei_max = 1.0
    ei_normalized = ei / ei_max
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-10:
        ucb_normalized = np.zeros_like(ucb)
    else:
        ucb_normalized = (ucb - ucb_min) / ucb_range
    
    # 4. Adaptive Weights
    # When f_best is close to 0 (minimum of rastrigin), rely more on EI (exploitation)
    # When f_best is large, rely more on UCB (exploration)
    if f_best < 1.0:
        ei_weight = 0.8
        ucb_weight = 0.2
    elif f_best < 10.0:
        ei_weight = 0.6
        ucb_weight = 0.4
    else:
        ei_weight = 0.4
        ucb_weight = 0.6
        
    # Combine normalized scores
    score = ei_weight * ei_normalized + ucb_weight * ucb_normalized
    
    # 5. Cost-awareness
    # Use (1 + cost) for stronger penalty on expensive evaluations
    # Cost is in [1, 2], so denominator is in [2, 3]
    cost_norm = 1.0 + cost
    score = score / cost_norm
    
    # 6. Uncertainty Bonus
    # Add a small bonus for high uncertainty regions to help escape local minima
    # Scale the bonus: higher when far from optimum
    if f_best > 1.0:
        bonus_scale = 0.1
    else:
        bonus_scale = 0.05
        
    uncertainty_bonus = bonus_scale * sigma / (1.0 + sigma)
    score = score + uncertainty_bonus / cost_norm
    
    # Check time budget - return early if needed
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Safety margin before 300s limit
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
