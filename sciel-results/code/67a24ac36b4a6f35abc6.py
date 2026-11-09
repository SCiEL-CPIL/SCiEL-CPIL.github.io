"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes heuristic scores for items based on Pareto dominance in the prize-weight space.
    
    Args:
        prizes: numpy array of shape (5, 100), uniform in [0, 1].
        weights: numpy array of shape (5, 100), uniform in [0, 1].
        
    Returns:
        numpy array of shape (100,), where each entry is the heuristic score for an item.
    """
    n_constraints, n_items = prizes.shape
    
    # 1. Aggregate prizes to a scalar value per item
    # Assuming the total prize is the sum of the 5 components
    total_prizes = np.sum(prizes, axis=0)
    
    # 2. Calculate dominance counts
    # Item j dominates item i if:
    # total_prizes[j] >= total_prizes[i]
    # weights[:, j] <= weights[:, i] (element-wise)
    # AND at least one inequality is strict
    
    domination_counts = np.zeros(n_items, dtype=np.int32)
    
    # Pre-extract weights for faster access in loop if needed, 
    # but direct indexing is fine for N=100.
    
    for i in range(n_items):
        p_i = total_prizes[i]
        w_i = weights[:, i]
        
        for j in range(n_items):
            if i == j:
                continue
            
            p_j = total_prizes[j]
            w_j = weights[:, j]
            
            # Check dominance of j over i
            # 1. Prize condition
            if p_j < p_i:
                continue
            
            # 2. Weight condition (all weights of j must be <= weights of i)
            # Using np.all for vector comparison
            if not np.all(w_j <= w_i):
                continue
            
            # 3. Strictness condition
            # It must be strictly better in at least one aspect
            # If p_j == p_i, it must have strictly lower weight in at least one constraint
            # If p_j > p_i, it dominates even if weights are equal (or lower)
            if p_j > p_i:
                domination_counts[i] += 1
            else:
                # p_j == p_i
                if np.any(w_j < w_i):
                    domination_counts[i] += 1
    
    # 3. Compute heuristic
    # High prize and low domination count is good.
    # We use total_prizes / (1 + domination_count)
    # This ensures items on the Pareto front (count=0) get the highest relative score for their prize.
    # Items dominated by many others get heavily penalized.
    
    heuristic = total_prizes / (1.0 + domination_counts)
    
    # Ensure all values are positive (as per task description: non-positive become 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
