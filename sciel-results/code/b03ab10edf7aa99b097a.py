import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 320.0 # 300s limit, keep 20s margin

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
        # We want to maximize attractiveness for pairs that leave little space (small gap).
        # Using 1 / (gap + 1) creates a strong preference for tight fits.
        # This mimics a "Best Fit" heuristic which is effective for bin packing.
        
        gaps = capacity - sums[valid]
        
        # Calculate heuristic values
        # Add 1.0 to denominator to avoid division by zero and scale values to [0, 1]
        eta[valid] = 1.0 / (gaps + 1.0)
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
        
    return eta
