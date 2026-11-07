import numpy as np
import time

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Capacity per constraint is 25
    capacity = 25.0
    
    # For each item, compute the fraction of capacity consumed in each constraint
    # weight_fraction[c, i] = weights[c, i] / capacity
    weight_fractions = weights / capacity  # shape (num_constraints, num_items)
    
    # The bottleneck for each item is the constraint where it consumes the most capacity
    max_fraction = np.max(weight_fractions, axis=0)  # shape (num_items,)
    
    # Total prize for each item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)  # shape (num_items,)
    
    # Use a smooth surrogate for the worst-case (max) constraint usage
    # Soft-max via log-sum-exp: log(sum(exp(alpha * x))) / alpha
    # As alpha -> infinity, this approaches max(x)
    # Using a moderate alpha to get a smooth approximation that still captures
    # the worst-case constraint while being less sensitive to single-constraint outliers
    alpha = 10.0  # Controls how closely we approximate the max
    
    # Compute soft-max of weight fractions
    # To avoid numerical issues, subtract the max before exponentiating
    # soft_max = max + (1/alpha) * log(sum(exp(alpha * (x - max))))
    max_w = np.max(weight_fractions, axis=0, keepdims=True)  # shape (1, num_items)
    shifted = alpha * (weight_fractions - max_w)
    # Clip to prevent overflow in exp
    shifted = np.clip(shifted, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=0))
    soft_max = max_w.squeeze() + (1.0 / alpha) * log_sum_exp  # shape (num_items,)
    
    # Heuristic: total prize divided by the soft-max capacity fraction
    # This rewards items with high total prize and low worst-case resource usage
    eps = 1e-9
    heuristic = total_prize / (soft_max + eps)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
