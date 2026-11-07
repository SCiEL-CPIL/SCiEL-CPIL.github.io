import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 320.0  # 300s limit, keep 20s margin

def heuristics(demands, capacity):
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones((len(demands), len(demands))) * 1e-9

    n = len(demands)
    # Convert to float array to be safe
    d = np.array(demands, dtype=np.float64)
    
    # Calculate sums
    sums = d[:, None] + d[None, :]
    
    # Create mask for valid pairs (fit in capacity)
    valid = sums <= capacity
    
    # Initialize eta with small positive values for invalid entries
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if np.any(valid):
        # Calculate gap for valid pairs
        gaps = capacity - sums[valid]
        
        # Primary signal: inverse gap with minimal damping (sharp best-fit preference)
        # Use 1/(gap + epsilon) where epsilon is very small to create strong peak at gap=0
        # This makes tight fits much more attractive than loose fits
        inv_gap = 1.0 / (gaps + 0.01)
        
        # Secondary signal: fill ratio with higher exponent to emphasize full bins
        ratio = sums[valid] / capacity
        fill_score = ratio ** 5.0
        
        # Tertiary signal: scarcity-based centrality
        # For each item j, count how many items i can fit with j (degree in compatibility graph)
        # Items with fewer valid partners are "scarce" and should be prioritized
        # This is a column-wise degree (how many items can precede j)
        degree_j = np.sum(valid, axis=0)  # shape (n,)
        # Scarcity score: higher for items with fewer valid predecessors
        # Use inverse degree with smoothing to avoid division by zero
        scarcity_j = 1.0 / (degree_j + 1.0)  # shape (n,)
        
        # For each valid pair (i,j), the scarcity of j
        # valid is (n,n) boolean, scarcity_j is (n,)
        scarcity_matrix = np.broadcast_to(scarcity_j[None, :], (n, n)).copy()
        
        # Blend: product of inv_gap, fill_score, and scarcity
        eta_valid = inv_gap * fill_score * scarcity_matrix[valid]
        
        # Normalize to [0, 1] range for consistency
        eta_max = np.max(eta_valid)
        if eta_max > 0 and np.isfinite(eta_max):
            eta_valid = eta_valid / eta_max
        
        eta[valid] = eta_valid
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
    
    return eta
