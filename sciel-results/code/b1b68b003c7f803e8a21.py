"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
        prizes: numpy array of shape (5, 100)
        weights: numpy array of shape (5, 100)
        
    Returns:
        numpy array of shape (100,) with heuristic scores.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Shape validation (defensive)
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback to simple heuristic if shapes are unexpected
        # Assuming the prompt guarantees shapes, but just in case
        try:
            n_items = prizes.shape[1]
            return np.ones(n_items)
        except:
            return np.ones(100)
            
    # 1. Calculate total prize per item (sum over constraints/dimensions)
    # V_i = sum_c prizes[c, i]
    total_prizes = np.sum(prizes, axis=0)
    
    # 2. Calculate total weight per constraint
    # W_c = sum_i weights[c, i]
    total_weights_per_constraint = np.sum(weights, axis=1)
    
    # 3. Calculate tightness factors
    # Capacity is 25 for all constraints.
    # If total weight > capacity, the constraint is tight.
    # We use a factor that increases the penalty for weights in tight constraints.
    capacity = 25.0
    # Avoid division by zero or extreme values
    tightness = np.maximum(1.0, total_weights_per_constraint / capacity)
    
    # 4. Calculate effective weight per item
    # E_i = sum_c weights[c, i] * tightness[c]
    # weights.T is (100, 5), tightness is (5,)
    effective_weights = np.dot(weights.T, tightness)
    
    # 5. Calculate heuristic
    # Heuristic is Value / Effective Weight
    # Add epsilon to avoid division by zero
    eps = 1e-9
    heur = total_prizes / (effective_weights + eps)
    
    # Ensure all values are positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    # Deadline guard (unlikely to trigger for 100 items)
    if time.time() - start_time > 320:
         return np.ones(100)
         
    return heur
# EVOLVE-BLOCK-END
