"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    Uses a geometric-mean per-constraint ratio combined with feasibility-aware scaling.
    
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
        try:
            n_items = prizes.shape[1]
            return np.ones(n_items)
        except:
            return np.ones(100)
            
    capacity = 25.0
    n_constraints = 5
    n_items = 100
    
    # Per-constraint prize and weight for each item
    # prizes: (5, 100), weights: (5, 100)
    
    # 1. Per-constraint ratio: prize[c,i] / (weight[c,i] + eps)
    # This captures the "bang per buck" in each constraint dimension
    eps = 1e-9
    per_constraint_ratio = prizes / (weights + eps)  # (5, 100)
    
    # 2. Geometric mean across constraints for each item
    # This penalizes items that are bad in ANY constraint (multiplicative)
    # rather than just penalizing by aggregate weight
    geo_ratio = np.exp(np.mean(np.log(np.maximum(per_constraint_ratio, eps)), axis=0))  # (100,)
    
    # 3. Feasibility-aware scaling: estimate how "expensive" each item is
    # relative to the tightest constraint it affects
    # For each item, find the maximum weight fraction across constraints
    weight_fraction = weights / capacity  # (5, 100), fraction of capacity used per constraint
    max_weight_fraction = np.max(weight_fraction, axis=0)  # (100,)
    
    # Items that use a large fraction of any single constraint are harder to fit
    # Scale down items with high max weight fraction
    feasibility_penalty = 1.0 / (1.0 + max_weight_fraction)  # (100,)
    
    # 4. Combine: geometric ratio weighted by feasibility
    heur = geo_ratio * feasibility_penalty
    
    # 5. Also incorporate total prize as a secondary signal
    # Normalize total prizes to [0, 1] range for blending
    total_prizes = np.sum(prizes, axis=0)  # (100,)
    tp_min = np.min(total_prizes)
    tp_max = np.max(total_prizes)
    if tp_max > tp_min:
        total_prizes_norm = (total_prizes - tp_min) / (tp_max - tp_min)
    else:
        total_prizes_norm = np.ones(n_items) * 0.5
    
    # Blend: 70% geometric-feasibility signal, 30% normalized total prize
    heur = 0.7 * heur + 0.3 * total_prizes_norm * np.max(heur)
    
    # Ensure all values are positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    # Deadline guard
    if time.time() - start_time > 320:
         return np.ones(100)
         
    return heur
# EVOLVE-BLOCK-END
