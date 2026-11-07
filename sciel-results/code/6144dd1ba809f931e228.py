"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    # Ensure inputs are float arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # prizes: (5, 100), weights: (5, 100)
    
    epsilon = 1e-9
    capacity = 25.0
    
    # Per-constraint ratio: prizes[i,j] / weights[i,j]
    # Add epsilon to weights to avoid division by zero
    safe_weights = weights + epsilon
    per_constraint_ratio = prizes / safe_weights  # (5, 100)
    
    # 1. Harmonic mean across constraints: 1 / mean(1/ratio_i)
    # This penalizes items with low ratio in any constraint
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_ratio = 1.0 / per_constraint_ratio
    harmonic_mean_ratio = 1.0 / (np.mean(inv_ratio, axis=0) + epsilon)  # (100,)
    
    # 2. Soft-min of per-constraint ratios using log-sum-exp
    # This creates a smooth penalty for the bottleneck constraint
    T = 0.05
    neg_ratio_scaled = -per_constraint_ratio / T
    # Numerical stability: subtract max before exp
    max_neg = neg_ratio_scaled.max(axis=0, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(neg_ratio_scaled - max_neg), axis=0)) + max_neg.squeeze(0)
    soft_min_ratio = -log_sum_exp * T  # (100,)
    
    # 3. Geometric slack factor: exp(mean(log(1 - w_i/capacity)))
    # Rewards items that are light across all constraints
    slack_factors = 1 - weights / capacity
    slack_factors = np.clip(slack_factors, epsilon, 1)
    log_slack = np.log(slack_factors)
    geometric_slack = np.exp(np.mean(log_slack, axis=0))  # (100,)
    
    # Combine measures:
    # Base: Harmonic mean (balanced efficiency)
    # Penalty: Soft-min (bottleneck awareness)
    # Bonus: Geometric slack (lightness reward)
    
    # Exponents tuned for balance
    bottleneck_factor = np.power(soft_min_ratio + epsilon, 0.7)
    slack_factor = np.power(geometric_slack, 0.4)
    
    heuristic = harmonic_mean_ratio * bottleneck_factor * slack_factor
    
    # Apply power to sharpen selection (higher power = more greedy)
    heuristic = np.power(heuristic + epsilon, 1.8)
    
    # Ensure positivity and handle non-finite values
    heuristic = np.where(heuristic > 0, heuristic, epsilon)
    heuristic = np.where(np.isfinite(heuristic), heuristic, epsilon)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Verify shape
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
    
    return heuristic
# EVOLVE-BLOCK-END
