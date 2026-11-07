"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 295.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're getting close to the 300s limit, return safe scores
    if time.time() - _start_time > _TIMEOUT:
        return np.ones_like(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle any remaining numerical issues
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Cost-aware: divide by cost to account for expensive evaluations
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute multiple acquisition functions and combine them
    # 1. Cost-aware EI
    ei_score = ei / cost_safe
    
    # 2. Cost-aware UCB with adaptive beta
    # Use a slightly higher beta for better exploration in early stages
    beta = 1.5 + 0.5 * np.exp(-np.mean(sigma_safe))  # Adaptive beta
    ucb = -mu + beta * sigma_safe
    ucb_score = ucb / cost_safe
    
    # 3. Cost-aware PI (Probability of Improvement)
    pi = norm.cdf(z)
    pi_score = pi / cost_safe
    
    # 4. Cost-aware lower confidence bound (LCB) - for minimization
    lcb = mu + beta * sigma_safe
    lcb_score = -lcb / cost_safe  # Negate because we want to minimize
    
    # 5. Cost-aware expected improvement with a bonus for low uncertainty
    uncertainty_bonus = np.log1p(sigma_safe)
    ei_with_bonus = ei + 0.1 * uncertainty_bonus
    ei_bonus_score = ei_with_bonus / cost_safe
    
    # Combine all scores with weights that emphasize exploration
    # Weight: EI (0.3), UCB (0.25), PI (0.15), LCB (0.15), EI+bonus (0.15)
    score = (0.3 * ei_score + 
             0.25 * ucb_score + 
             0.15 * pi_score + 
             0.15 * lcb_score + 
             0.15 * ei_bonus_score)
    
    # Normalize scores to [0, 1] for better numerical stability
    score_min = np.min(score)
    score_max = np.max(score)
    score_range = score_max - score_min
    if score_range > 1e-10:
        score = (score - score_min) / score_range
    else:
        score = np.ones_like(score) * 0.5
    
    # Add a small random perturbation to break ties and encourage diversity
    # This helps escape local optima in the acquisition function landscape
    perturbation = np.random.uniform(0, 0.01, size=score.shape)
    score = score + perturbation
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
