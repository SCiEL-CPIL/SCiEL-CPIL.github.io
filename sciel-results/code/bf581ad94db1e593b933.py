"""Cost-aware UCB acquisition with budget-scaled exploration for BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 315.0  # 15s margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are too close to the limit, return a safe fallback
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros(len(mu))
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Budget-aware exploration: scale beta by remaining budget ratio
    # cost is the cumulative cost so far; total budget is 30
    # Remaining budget ratio: (30 - mean(cost)) / 30
    # This makes exploration stronger early (high remaining budget) and weaker late
    remaining_ratio = np.clip((30.0 - np.mean(cost)) / 30.0, 0.0, 1.0)
    
    # Beta scales with remaining budget: more exploration when budget is plentiful
    # Base beta of 2.0, scaled down as budget depletes
    beta = 2.0 * remaining_ratio + 0.1
    
    # UCB on reward: reward = -mu (lower objective is better, so higher reward)
    # UCB_reward = -mu + beta * sigma
    ucb_reward = -mu + beta * sigma_safe
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: UCB_reward / sqrt(cost)
    # Using sqrt(cost) instead of cost gives a softer penalty for expensive points,
    # which helps balance exploration in high-cost regions
    scores = ucb_reward / np.sqrt(cost_safe)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
