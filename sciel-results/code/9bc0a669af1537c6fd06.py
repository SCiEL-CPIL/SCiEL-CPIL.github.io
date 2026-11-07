# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Lower Confidence Bound with adaptive beta, plus cost normalization.
    
    Strategy:
    - Compute EI and LCB scores
    - Blend them with an adaptive weight that favors exploration early and
      exploitation later (based on how close mu is to f_best)
    - Normalize by cost to account for evaluation expense
    - Add a small uncertainty bonus to encourage exploration in high-sigma regions
    
    This structural change from pure EI/cost to a blended EI+LCB approach
    with uncertainty-aware exploration should find the global minimum faster.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values and safe ranges
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma to avoid division by zero or numerical instability
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Clip cost to avoid division by zero
    cost = np.clip(cost, 1e-6, 100.0)
    
    # Handle cases where sigma is very small
    small_sigma_mask = sigma < 1e-5
    
    # Compute Expected Improvement
    ei = np.zeros_like(mu)
    normal_sigma_mask = ~small_sigma_mask
    
    if np.any(normal_sigma_mask):
        mu_n = mu[normal_sigma_mask]
        sigma_n = sigma[normal_sigma_mask]
        
        z = (f_best - mu_n) / sigma_n
        z = np.clip(z, -10, 10)
        
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        ei_n = (f_best - mu_n) * Phi_z + sigma_n * phi_z
        ei[normal_sigma_mask] = ei_n
    
    # For small sigma, EI is approx max(f_best - mu, 0)
    ei[small_sigma_mask] = np.maximum(f_best - mu[small_sigma_mask], 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Compute Lower Confidence Bound
    # Adaptive beta: larger when uncertainty is high relative to potential improvement
    # Use a simple heuristic: beta = 1.0 + 0.5 * (sigma / max(f_best - min(mu), 1e-6))
    mu_min = np.min(mu)
    range_scale = max(f_best - mu_min, 1e-6)
    
    # Adaptive beta: ranges from ~1.0 to ~3.0
    beta = 1.0 + 1.5 * (sigma / (sigma + range_scale))
    beta = np.clip(beta, 0.5, 3.0)
    
    lcb = mu - beta * sigma
    
    # Blend EI and LCB
    # When mu is close to f_best (exploiting), weight EI more
    # When mu is far from f_best (exploring), weight LCB more
    # Use a sigmoid-like blend based on normalized distance
    dist_to_best = np.maximum(f_best - mu, 0.0)
    # Normalize distance: 0 when at best, 1 when far
    dist_norm = dist_to_best / (dist_to_best + range_scale)
    
    # Weight for EI: higher when close to best (exploit)
    # Weight for LCB: higher when far from best (explore)
    w_ei = 0.5 + 0.5 * (1.0 - dist_norm)  # 0.5 to 1.0
    w_lcb = 1.0 - w_ei  # 0.0 to 0.5
    
    # Normalize LCB to be comparable to EI
    # LCB can be negative, EI is non-negative
    # Shift LCB so that its max is aligned with EI scale
    lcb_shift = np.maximum(lcb, 0.0)
    
    # Combine
    score = w_ei * ei + w_lcb * lcb_shift
    
    # Add a small uncertainty bonus to encourage exploration
    # This helps escape local minima
    uncertainty_bonus = 0.1 * sigma / (sigma + range_scale)
    score = score + uncertainty_bonus
    
    # Cost-aware normalization
    score = score / cost
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
