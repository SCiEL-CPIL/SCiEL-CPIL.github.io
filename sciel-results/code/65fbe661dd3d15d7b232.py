import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME, _DEADLINE
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Deadline guard: if we're running out of time, return simple EI/cost
    if time.time() > _DEADLINE - 2.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        score = np.maximum(score, 0.0)
        return score
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration term
    # Beta scales with remaining budget ratio for adaptive exploration
    # Total budget is 30, initial cost from 5 random points is ~5*1.5=7.5
    # So remaining budget = 30 - current_total_cost
    # We estimate current total cost from the mean of the cost array (all candidates have similar cost structure)
    # Actually, cost here is the cost of evaluating each candidate point
    # The total spent is tracked by the loop, but we can estimate progress
    
    # Use f_best as a proxy for progress: high f_best = early, low f_best = late
    # Also use the spread of sigma to gauge exploration needs
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_ratio = sigma_mean / (sigma_max + 1e-9)
    
    # Dynamic beta: higher when uncertainty is high (early) and lower when confident (late)
    # f_best scale: normalize by typical initial f_best for powell 4d on [-4,5]^4
    # Initial f_best is likely in range [0, ~10]
    f_best_norm = min(abs(f_best) / 10.0, 1.0)
    
    # Beta starts high (explore) and decreases (exploit)
    # Also modulate by sigma distribution: if GP is uncertain everywhere, explore more
    beta_base = 2.0 * (1.0 - f_best_norm) + 0.5
    beta = beta_base * (0.5 + 0.5 * sigma_ratio)
    
    ucb = mu + beta * sigma_safe
    
    # Cost-normalized blending
    # Weight EI/cost more heavily, with UCB as exploration supplement
    # The blend shifts from UCB-dominant (early) to EI-dominant (late)
    f_best_scale = max(abs(f_best), 1e-6)
    alpha = 1.0 / (1.0 + f_best_scale)
    
    cost_weight_ei = cost_safe ** (-alpha)
    cost_weight_ucb = cost_safe ** (-(alpha * 0.5))
    
    # Blend: early = more UCB, late = more EI
    blend_weight = 0.3 * (1.0 - f_best_norm)  # UCB weight
    
    score_ei = ei * cost_weight_ei
    score_ucb = np.maximum(-ucb, 0.0) * cost_weight_ucb  # Lower ucb = better, so negate
    
    score = (1.0 - blend_weight) * score_ei + blend_weight * score_ucb
    
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
