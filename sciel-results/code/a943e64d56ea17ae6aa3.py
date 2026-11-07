import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # 1. Total prize per item
    total_prizes = np.sum(prizes, axis=0) # (100,)
    
    # 2. Total weight per constraint
    total_weights_per_constraint = np.sum(weights, axis=1) # (5,)
    
    # 3. Scarcity factors
    capacity = 25.0
    scarcity = total_weights_per_constraint / capacity # (5,)
    
    # Normalize scarcity to mean 1 to keep scale comparable to baseline
    mean_scarcity = np.mean(scarcity)
    if mean_scarcity > 1e-9:
        scarcity_norm = scarcity / mean_scarcity
    else:
        scarcity_norm = np.ones_like(scarcity)
        
    # 4. Effective weight per item
    # weights.T is (100, 5), scarcity_norm is (5,)
    # Dot product gives (100,)
    effective_weights = np.dot(weights.T, scarcity_norm)
    
    # 5. Heuristic
    eps = 1e-9
    heur = total_prizes / (effective_weights + eps)
    
    # Ensure positive
    heur = np.maximum(heur, 1e-9)
    
    # Check deadline (unlikely to be hit)
    if time.time() > deadline:
         # Return a safe default if somehow slow
         return np.ones(100)
         
    return heur
