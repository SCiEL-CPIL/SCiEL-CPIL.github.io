import numpy as np
import time

# Pre-compute the RBF centers and features to minimize overhead during evaluation
# Grid of 21x21 points in [-1, 1]
grid = np.linspace(-1, 1, 21)
xx, yy = np.meshgrid(grid, grid, indexing='ij')
centers_x = xx.flatten()
centers_y = yy.flatten()
n_centers = len(centers_x)

# We will select 10 centers dynamically based on the data, but since we can't 
# change the structure of params (fixed length 10), we must map the 10 params 
# to specific centers. 
# Strategy: Use the first 10 centers in a space-filling pattern or just the first 10.
# Better Strategy: Since we don't know the function, a fixed set of 10 centers 
# might miss the action. However, we can't iterate over 441 centers with only 
# 10 params.
# 
# Alternative Strategy: Use a "Learnable" center approach? No, params are just scalars.
# 
# Let's stick to a fixed, well-distributed set of 10 centers.
# A 3x3 grid plus 2 extra points? Or just the first 10 of a sorted list?
# To maximize coverage, let's pick centers that are spread out.
# Indices for a roughly uniform distribution of 10 points in 21x21 grid.
# Row 0: 0, 10, 20
# Row 10: 21+0, 21+10, 21+20
# Row 20: 42+0, 42+10, 42+20
# Plus 2 in the middle?
# Let's just take every 44th index? 441/10 = 44.1
indices = np.linspace(0, n_centers - 1, 10, dtype=int)

cx = centers_x[indices]
cy = centers_y[indices]

# Pre-compute distances from any (x,y) to these 10 centers is cheap.
# But we can pre-compute nothing here because x,y change.
# However, we can define the function efficiently.

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Safety guard: if we exceed 140s, return a simple fallback
    # Though RBF is fast, BFGS might be slow if condition number is bad.
    # RBFs can be ill-conditioned if centers are close.
    # Let's use a wider bandwidth to avoid ill-conditioning.
    sigma = 0.5
    
    # Compute distances squared to the 10 selected centers
    # x, y are arrays of shape (N,)
    # cx, cy are arrays of shape (10,)
    
    # Efficient computation:
    # dx = x[:, None] - cx[None, :]  # (N, 10)
    # dy = y[:, None] - cy[None, :]  # (N, 10)
    # dist_sq = dx**2 + dy**2        # (N, 10)
    # features = np.exp(-dist_sq / (2 * sigma**2)) # (N, 10)
    # pred = features @ params
    
    # Vectorized approach:
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    dist_sq = dx * dx + dy * dy
    features = np.exp(-dist_sq / (2 * sigma * sigma))
    
    pred = features @ params
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to linear if something explodes
        return params[0] * x + params[1] * y + params[2]
        
    return pred
