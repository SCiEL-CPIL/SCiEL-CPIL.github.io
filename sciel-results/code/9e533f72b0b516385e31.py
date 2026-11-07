import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # prizes: (5, 100), weights: (5, 100)
    # 5 constraints, capacity 25 each

    # Ensure input is float
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Calculate efficiency ratios for each constraint and item
    # R[k, j] = prize[k, j] / weight[k, j]
    # Use epsilon to avoid division by zero
    epsilon = 1e-9
    safe_weights = np.where(weights > epsilon, weights, 1.0)
    ratios = prizes / safe_weights  # Shape (5, 100)

    # 1. Harmonic Mean of ratios for each item
    # H_j = n / sum(1/r_kj)
    # 1/r_kj = safe_weights[k, j] / prizes[k, j]
    
    denom = np.where(prizes > epsilon, prizes, 1.0)
    inv_ratios = safe_weights / denom
    
    sum_inv_ratios = np.sum(inv_ratios, axis=0) # (100,)
    
    # Harmonic Mean H_j
    H = np.where(sum_inv_ratios > epsilon, 5.0 / sum_inv_ratios, 1e6)

    # 2. Soft-Min-Ratio (Bottleneck) M_j
    # Use soft-min via log-sum-exp for smoothness
    # soft_min(x, t) = -t * log(sum(exp(-x_i/t)))
    # As t -> 0, this approaches min(x)
    # Tuning t_soft: lower t makes it closer to hard min (more penalty on worst constraint)
    # Higher t makes it smoother (less penalty)
    # Previous best used 0.1. Let's try a slightly lower temperature to be more sensitive to the bottleneck.
    t_soft = 0.05  # Reduced from 0.1 to emphasize the bottleneck more strongly
    neg_ratios = -ratios / t_soft
    # Clip to avoid overflow in exp
    neg_ratios = np.clip(neg_ratios, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(neg_ratios), axis=0))
    M = -t_soft * log_sum_exp  # (100,)

    # 3. Geometric Blending
    # eta = H^alpha * M^beta
    # Previous: alpha=1.5, beta=0.5
    # Hypothesis: Increasing the weight on the bottleneck (M) might help avoid infeasible-heavy packs
    # or better prioritize items that are good across ALL constraints.
    # Let's try alpha=1.2, beta=0.8 to give more weight to the soft-min bottleneck.
    # Or perhaps alpha=1.0, beta=1.0 (geometric mean).
    # Let's stick close to the previous success but tune towards bottleneck sensitivity.
    alpha = 1.2
    beta = 0.8
    
    H_safe = np.maximum(H, epsilon)
    M_safe = np.maximum(M, epsilon)
    
    heuristic = np.power(H_safe, alpha) * np.power(M_safe, beta)
    
    # 4. Normalization
    # Normalize to mean 1 to match pheromone scale
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones_like(heuristic)
        
    # 5. Safety Checks
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
        
    if time.time() > deadline:
        return heuristic
        
    return heuristic
