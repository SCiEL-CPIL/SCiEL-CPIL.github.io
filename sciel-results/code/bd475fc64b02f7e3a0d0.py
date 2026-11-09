import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the MCKP ACO solver.
    Uses prize divided by the maximum weight across constraints.
    This favors items that are 'light' in the bottleneck constraint,
    preserving feasibility across all constraints.
    """
    start_time = time.time()
    deadline = 325.0  # 300s limit + margin, but we are fast so this is just a guard
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Calculate max weight for each item across all constraints
    # weights shape is (5, 100)
    max_weights = np.max(weights, axis=0)
    
    # Calculate heuristic: prize / max_weight
    # Add epsilon to avoid division by zero
    heuristic = prizes / (max_weights + 1e-9)
    
    # Ensure no non-finite values
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Check time (should be negligible)
    if time.time() - start_time > deadline:
        # Fallback to a simple uniform heuristic if we are somehow slow (should not happen)
        return np.ones(100)
        
    return heuristic
