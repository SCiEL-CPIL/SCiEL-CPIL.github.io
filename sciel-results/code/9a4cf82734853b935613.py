import numpy as np

def heuristics(demands, capacity):
    try:
        n = len(demands)
        d = np.array(demands, dtype=np.float64)
        pair = d[:, None] + d[None, :]
        
        residual = capacity - pair
        valid = residual >= 0
        
        h = np.full((n, n), 1e-9, dtype=np.float64)
        
        # Use 1 / (residual + 1) to avoid division by zero and provide smooth decay
        # This strongly prefers smaller gaps but doesn't explode for perfect fits
        h[valid] = 1.0 / (residual[valid] + 1.0)
        
        # Ensure no negative values or NaNs
        h[~np.isfinite(h)] = 1e-9
        h[h < 0] = 1e-9
        
        return h
    except Exception:
        # Fallback to a simple heuristic if something goes wrong
        n = len(demands)
        return np.full((n, n), 1e-9)
