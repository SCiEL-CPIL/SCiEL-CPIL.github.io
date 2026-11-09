import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the MCKP ACO solver.
    Uses prize divided by a soft-max of weights across constraints.
    This provides a smooth surrogate for the hard max, better differentiating
    items that are light in all constraints vs. those that are heavy in one.
    """
    start_time = time.time()
    deadline = 325.0  # 300s limit + margin, but we are fast so this is just a guard
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Calculate soft-max of weights for each item across constraints
    # Using log-sum-exp for numerical stability
    # weights shape is (5, 100)
    # For each item (column), compute soft-max over the 5 constraints (rows)
    
    # Temperature parameter for soft-max: smaller = closer to hard max
    # Previous T=0.1 scored 0.9879. Seed (likely hard max) scored 1.0.
    # Hypothesis: Smaller T (sharper max) will improve score by better
    # identifying the bottleneck constraint.
    temperature = 0.05
    
    # Normalize weights to [0, 1] range for better soft-max behavior
    # Actually, let's work with the raw weights but scale appropriately
    
    # Soft-max computation:
    # soft_max_j = sum_i exp(w_ij / T) * T  (this is the log-sum-exp scaled by T)
    # More precisely: T * log(sum_i exp(w_ij / T))
    
    # For numerical stability, subtract max before exp
    max_w = np.max(weights, axis=0, keepdims=True)  # shape (1, 100)
    scaled_weights = (weights - max_w) / temperature  # shape (5, 100)
    
    # Clip to avoid overflow in exp
    scaled_weights = np.clip(scaled_weights, -50, 50)
    
    exp_weights = np.exp(scaled_weights)  # shape (5, 100)
    
    # Log-sum-exp: log(sum(exp))
    log_sum_exp = np.log(np.sum(exp_weights, axis=0))  # shape (100,)
    
    # Soft-max value: T * log_sum_exp + max_w
    # This gives a smooth approximation to the max
    soft_max_weights = temperature * log_sum_exp + np.squeeze(max_w, axis=0)  # shape (100,)
    
    # Calculate heuristic: prize / soft_max_weight
    # Add epsilon to avoid division by zero
    heuristic = prizes / (soft_max_weights + 1e-9)
    
    # Ensure no non-finite values
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Check time (should be negligible)
    if time.time() - start_time > deadline:
        # Fallback to a simple uniform heuristic if we are somehow slow (should not happen)
        return np.ones(100)
        
    return heuristic
