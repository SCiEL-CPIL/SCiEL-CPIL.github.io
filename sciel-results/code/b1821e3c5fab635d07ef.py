import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin

    # Ensure prizes is 1D
    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes

    # Ensure weights is 2D (K, N)
    w = weights
    K, N = w.shape
    
    # Calculate total weight per constraint to estimate scarcity
    # Capacity is 25
    total_weights = w.sum(axis=1)  # Shape (K,)
    capacity = 25.0
    # Scarcity factor: higher if constraint is tighter
    # Avoid division by zero or negative if total > capacity? 
    # If total > capacity, scarcity is high.
    # S_k = max(1.0, total_weights[k] / capacity)
    # Or simply use the ratio to scale the weights in the LSE
    scarcity = total_weights / capacity
    # Clip scarcity to [0.5, 2.0] to avoid extreme values if data is noisy
    scarcity = np.clip(scarcity, 0.5, 2.0)
    
    # Calculate Log-Sum-Exp based cost for each item
    # C_i = (1/beta) * log(sum_k exp(beta * S_k * w_ki))
    
    beta = 8.0  # Tuned parameter for soft-min/max
    
    # w is (K, N), scarcity is (K,)
    # We want to compute LSE along axis 0 for each column (item)
    # scaled_weights[k, i] = scarcity[k] * w[k, i]
    scaled_w = w * scarcity[:, np.newaxis]
    
    # Numerically stable LSE
    # max_val = max_k (beta * scaled_w[k, i])
    max_val = np.max(beta * scaled_w, axis=0)  # Shape (N,)
    
    # exp_terms = exp(beta * scaled_w - max_val)
    # sum_exp = sum_k exp_terms
    # lse = (max_val + log(sum_exp)) / beta
    
    exp_terms = np.exp(beta * scaled_w - max_val[np.newaxis, :])
    sum_exp = np.sum(exp_terms, axis=0)
    
    # Handle case where sum_exp is 0 (should not happen with exp of real numbers, but safe)
    sum_exp = np.maximum(sum_exp, 1e-300)
    
    lse_cost = (max_val + np.log(sum_exp)) / beta
    
    # Ensure cost is positive
    lse_cost = np.maximum(lse_cost, 1e-9)
    
    # Heuristic: Prize / Cost
    h = p / lse_cost
    
    # Ensure positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return h

    return h
