import numpy as np

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape mismatches
    # Expected: (5, 100)
    # Error suggests: (100,)
    
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # Now both should be 2D. 
    # If they have different number of rows, we might need to handle that, 
    # but typically they should match or be broadcastable. 
    # For safety, let's ensure they have the same number of rows (constraints).
    # If one is (1, 100) and other is (5, 100), we might need to expand the single constraint.
    # However, the task says 5 constraints. If input is 1D, it's ambiguous. 
    # Let's assume if 1D, it's a single constraint instance or the data is just flat.
    # Given the task description "3 fixed instances with 100 items and 5 constraints", 
    # and the error, it's highly probable the input is 1D for some reason (maybe a bug in data loading or a specific test case).
    # Let's just reshape to 2D and proceed. If shapes don't match for broadcasting, we'll deal with it, 
    # but min/max along axis 1 will work on (1, 100).
    
    # Normalize weights per constraint to [0, 1]
    w_min = np.min(weights, axis=1, keepdims=True)
    w_max = np.max(weights, axis=1, keepdims=True)
    w_range = w_max - w_min
    # Avoid division by zero
    w_range = np.where(w_range < 1e-12, 1.0, w_range)
    weights_norm = (weights - w_min) / w_range  # shape (num_constraints, 100)
    
    # For each item, find the maximum normalized weight across constraints
    # This represents the "worst-case" resource consumption relative to range
    max_norm_weight = np.max(weights_norm, axis=0)  # shape (100,)
    
    # Normalize prizes per constraint to [0, 1] as well
    p_min = np.min(prizes, axis=1, keepdims=True)
    p_max = np.max(prizes, axis=1, keepdims=True)
    p_range = p_max - p_min
    p_range = np.where(p_range < 1e-12, 1.0, p_range)
    prizes_norm = (prizes - p_min) / p_range  # shape (num_constraints, 100)
    
    # Sum normalized prizes across constraints to get total value
    total_prize_norm = np.sum(prizes_norm, axis=0)  # shape (100,)
    
    # Heuristic: product of normalized prize and inverse of max normalized weight
    # Add epsilon to avoid division by zero
    eps = 1e-9
    heuristic = total_prize_norm / (max_norm_weight + eps)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
