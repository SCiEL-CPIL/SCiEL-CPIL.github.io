import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 5.0 # 5s is plenty for N=50
    
    # Ensure input is a numpy array
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Add epsilon to avoid division by zero
    eps = 1e-9
    d_safe = d + eps
    
    # Apply power law heuristic
    # Beta = 2.0 is a standard choice to emphasize short edges
    beta = 2.0
    H = 1.0 / (d_safe ** beta)
    
    # Check for non-finite values (should not happen with d_safe > 0)
    # But just in case
    H[~np.isfinite(H)] = 1e-9
    
    # Ensure positive
    H[H <= 0] = 1e-9
    
    # Time check (unlikely to trigger)
    if time.time() > deadline:
         # If we somehow took too long, return a safe fallback
         # Though this is extremely unlikely for N=50
         pass
        
    return H
