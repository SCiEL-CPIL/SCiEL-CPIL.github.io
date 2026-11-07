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
        # Use inverse-gap heuristic: higher attractiveness for tighter fits
        # gap = capacity - sum
        # heuristic = 1 / (gap + 1)
        # This mimics Best-Fit Decreasing behavior
        valid_sums = sums[valid]
        gaps = capacity - valid_sums
        # Avoid division by zero, though gap >= 0 for valid pairs
        # Adding 1 ensures denominator is at least 1
        values = 1.0 / (gaps + 1.0)
        
        eta[valid] = values
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
        
    return eta
