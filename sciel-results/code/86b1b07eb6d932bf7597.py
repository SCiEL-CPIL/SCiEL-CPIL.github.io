import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0  # Safety margin before 330s kill
    
    # Handle potential shape issues
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)
    else:
        P = prizes
        
    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1)
        
    # W is (5, 100), P is (100,)
    n_constraints, n_items = W.shape
    
    # Per-constraint efficiency: prize / weight for each constraint
    # eff[c, i] = P[i] / W[c, i]
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = P[None, :] / (W + 1e-9)
    eff = np.where(np.isfinite(eff), eff, 0.0)
    
    # For each constraint, find the top-k most efficient items
    # This represents the "benchmark" efficiency level
    k = max(1, n_items // 5)  # top 20% of items
    top_k_eff = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_k_eff[c] = np.partition(eff[c], -k)[-k].mean()
    
    # Scarcity weight: constraints with higher top-k efficiency are "scarcer"
    # because the best items use them most efficiently, meaning they bind first
    # Normalize scarcity weights
    scarcity = top_k_eff / (top_k_eff.sum() + 1e-9)
    
    # Marginal utility: for each item, compute how much it contributes relative to scarcity
    # An item is valuable if it's efficient in constraints that are scarce
    # marginal_util[i] = sum_c (scarcity[c] * eff[c, i])
    marginal_util = (scarcity[:, None] * eff).sum(axis=0)
    
    # Also compute a base efficiency ratio as fallback
    W_sum = W.sum(axis=0)
    base_eff = P / (W_sum + 1e-9)
    
    # Combine: use marginal utility scaled by base efficiency
    # This rewards items that are both generally efficient AND efficient in scarce constraints
    H = marginal_util * base_eff
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    # Deadline guard: if we're close to the limit, return immediately
    if time.time() > deadline - 5.0:
        return H
    
    return H
