import numpy as np

def heuristics(prizes, weights):
    # Handle potential shape issues
    # If prizes is (5, 100), sum to (100,)
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)
    else:
        P = prizes
        
    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1) # unlikely, but safe
        
    # W is (5, 100)
    n_constraints, n_items = W.shape
    capacity = 25.0
    
    # Constraint-specific scarcity premium:
    # For each item, identify the most binding constraint (largest fraction of total weight)
    # and scale its prize-to-weight ratio by the inverse of remaining capacity for that constraint
    
    # Compute per-constraint efficiency: P / W[c, i] for each constraint c and item i
    # This gives the value per unit of resource c for item i
    W_safe = np.maximum(W, 1e-9)
    eff_per_constraint = P[None, :] / W_safe  # shape (5, 100)
    
    # For each item, find the constraint where its weight is largest fraction of total weight
    W_sum = np.sum(W, axis=0)  # (100,)
    W_fraction = W / (W_sum[None, :] + 1e-9)  # (5, 100)
    
    # The most binding constraint for each item is where W_fraction is largest
    most_binding_idx = np.argmax(W_fraction, axis=0)  # (100,)
    
    # For each item, get the efficiency in its most binding constraint
    item_indices = np.arange(n_items)
    eff_most_binding = eff_per_constraint[most_binding_idx, item_indices]  # (100,)
    
    # Compute remaining capacity fraction for the most binding constraint
    # Total weight used by all items in constraint c is sum(W[c, :])
    # Remaining capacity = capacity - sum(W[c, :])
    total_weight_per_constraint = np.sum(W, axis=1)  # (5,)
    remaining_capacity = np.maximum(capacity - total_weight_per_constraint, 1e-9)  # (5,)
    
    # For each item, the remaining capacity fraction for its most binding constraint
    remaining_frac_most_binding = remaining_capacity[most_binding_idx]  # (100,)
    
    # Scarcity premium: 1 / remaining capacity fraction
    # Items whose most binding constraint is more scarce get higher premium
    scarcity_premium = 1.0 / (remaining_frac_most_binding + 1e-9)  # (100,)
    
    # Scale the most-binding efficiency by scarcity premium
    H_scarcity = eff_most_binding * scarcity_premium
    
    # Also compute a baseline aggregate efficiency
    H_baseline = P / (W_sum + 1e-9)
    
    # Blend: weight the scarcity-aware heuristic more heavily
    alpha = 0.7
    H = alpha * H_scarcity + (1 - alpha) * H_baseline
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
