import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # prizes: (5, 100), weights: (5, 100)
    # 5 constraints, capacity 25 each
    
    # 1. Calculate total prize and total weight per item
    # P_j = sum of prizes across constraints (assuming total prize is sum of components)
    P = np.sum(prizes, axis=0)  # (100,)
    
    # W_j = sum of weights across constraints (total resource consumption)
    W = np.sum(weights, axis=0)  # (100,)
    
    # M_j = max weight in any single constraint (bottleneck resource usage)
    M = np.max(weights, axis=0)  # (100,)
    
    epsilon = 1e-9
    capacity = 25.0
    
    # 2. Estimate Lagrange multiplier (shadow price of resources)
    # Efficiency ratio R_j = P_j / W_j
    # We use a safe division
    safe_W = np.where(W > epsilon, W, 1.0)
    R = P / safe_W
    
    # The "cutoff" efficiency is roughly the median, as we can only select ~50% of items by weight
    # Using median makes it robust to outliers
    lambda_est = np.median(R)
    
    # If lambda is 0 or very small, fall back to mean or 1.0
    if not np.isfinite(lambda_est) or lambda_est < epsilon:
        lambda_est = np.mean(R)
        if not np.isfinite(lambda_est) or lambda_est < epsilon:
            lambda_est = 1.0
            
    # 3. Calculate Lagrangian Value (Surplus)
    # V_j = P_j - lambda * W_j
    V = P - lambda_est * W
    
    # 4. Incorporate Bottleneck Penalty
    # Items that consume a large fraction of a single constraint are "blockers"
    # Penalty factor: 1 - alpha * (M_j / Capacity)
    # If M_j is close to Capacity, the item is huge and blocks that constraint significantly
    # We use alpha = 0.5 to soften the penalty
    bottleneck_frac = M / capacity
    # Clip to [0, 1] to be safe, though weights are in [0,1] and cap is 25, so frac is in [0, 0.04]
    # Wait, weights are uniform in [0, 1]. Capacity is 25.
    # So M_j is at most 1.0. M_j / 25 is at most 0.04.
    # The penalty will be very small: 1 - 0.5 * 0.04 = 0.98.
    # This suggests the bottleneck penalty is negligible for these specific instance parameters (weights << capacity).
    # In this regime, the "bottleneck" is not a single item blocking a constraint, but the aggregate.
    # So the Lagrangian term P - lambda*W is the dominant factor.
    
    # Let's re-evaluate the "bottleneck" concept.
    # If weights are small [0,1], no single item is a bottleneck.
    # The constraint is aggregate.
    # So the Lagrangian $P_j - \lambda \sum w_{kj}$ is the correct structural model.
    
    # 5. Construct Heuristic
    # We want items with high Surplus V_j to have high heuristic.
    # Items with negative Surplus should have low heuristic.
    # To avoid zeroing out completely (which might stop exploration if pheromone is low),
    # we shift and scale.
    
    # Normalize V to be positive.
    # min_V = np.min(V)
    # V_norm = V - min_V
    # But we want to preserve the "goodness" relative to 0.
    
    # Let's use: eta_j = max(epsilon, V_j)
    # This gives 0 (or epsilon) to items that are worse than average efficiency.
    # This is a strong filter.
    
    # Alternative: Softplus or exponential?
    # eta_j = exp(V_j / scale) ?
    # If V_j is large positive, eta is large.
    # If V_j is negative, eta is small.
    
    # Let's try a linear mapping that keeps positive values for "above average" items
    # and a small floor for others.
    
    heuristic = np.maximum(V, epsilon)
    
    # Check if all are epsilon (degenerate case)
    if np.allclose(heuristic, epsilon):
        # Fallback to simple density
        heuristic = P / safe_W
        heuristic = np.maximum(heuristic, epsilon)
        
    # Ensure finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    # Optional: Normalize to mean 1? 
    # ACO probability is proportional to tau * eta. 
    # The absolute scale of eta matters relative to tau.
    # Initial tau is 1. 
    # If eta is ~1, prob is ~1.
    # If eta is ~100, prob is ~100.
    # Let's normalize to have mean 1 to be safe with pheromone scaling.
    
    mean_h = np.mean(heuristic)
    if mean_h > epsilon:
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones_like(heuristic)
        
    # Final safety checks
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
        
    if time.time() > deadline:
        return heuristic
        
    return heuristic
