"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    Uses a combination of product of sizes (favoring large items) and
    inverse waste squared (favoring tight fits) to maximize bin utilization.
    """
    start_time = time.time()
    
    # Convert to numpy array for vectorized operations
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate sums to check if items fit together
    sums = d[:, None] + d[None, :]
    
    # Calculate products to measure "compatibility" (favoring large-large pairs)
    products = d[:, None] * d[None, :]
    
    # Calculate waste (how much space is left in the bin after placing both items)
    # Waste is maximized when sum is small, minimized when sum is close to capacity
    waste = capacity - sums
    
    # Condition: items must fit together
    fits = sums <= capacity
    
    # For fitting pairs, compute heuristic as product * (1 / (waste + epsilon)^2)
    # This strongly favors pairs that are both large (high product) and tight (low waste)
    # The squared inverse waste creates a sharp peak for near-perfect fits
    
    # Use a small epsilon to avoid division by zero when waste is exactly 0
    epsilon = 1e-6
    waste_safe = np.where(fits, waste + epsilon, 1.0)  # Only compute for fitting pairs
    
    # Compute the heuristic values for fitting pairs
    # product * (1 / waste_safe^2)
    H_fitting = products / (waste_safe ** 2)
    
    # Set non-fitting pairs to 1e-9
    H = np.where(fits, H_fitting, 1e-9)
    
    # Ensure shape is correct
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
        
    # Check for non-finite values and sanitize if needed
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
