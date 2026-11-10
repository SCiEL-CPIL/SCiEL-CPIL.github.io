import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy: Direct heuristic based on maximizing bin fill efficiency.
    The attractiveness of placing item j after item i is defined by:
    1. Feasibility: Items must fit together (sum <= capacity).
    2. Efficiency: Pairs that leave less empty space (waste) are preferred.
    3. Magnitude: Pairs involving larger items are preferred to ensure large
       items are placed in high-utility positions.
    
    Formula:
    H[i, j] = (d[i] * d[j]) / (capacity - d[i] - d[j] + 1)^2  if d[i] + d[j] <= capacity
    H[i, j] = 1e-9                                            otherwise
    """
    start_time = time.time()
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate sums and products for all pairs
    sums = d[:, None] + d[None, :]
    products = d[:, None] * d[None, :]
    
    # Determine which pairs fit in the bin
    fits = sums <= capacity
    
    # Calculate waste for fitting pairs
    # waste = capacity - sum
    # We add 1.0 to the waste in the denominator to avoid division by zero
    # and to provide a stable scale (discrete unit logic).
    # For non-fitting pairs, we don't care about the value, but we mask it later.
    # To prevent warnings or errors during computation of the raw score,
    # we can compute the denominator safely.
    
    # raw_denom = (capacity - sums + 1.0)
    # For non-fitting pairs, sums > capacity, so capacity - sums is negative.
    # (capacity - sums + 1.0) could be small negative, zero, or positive.
    # Squaring it makes it positive.
    # But we want to mask them out anyway.
    
    # To be safe and fast, compute the term for all, then mask.
    # If sums > capacity, we set the final result to 1e-9.
    
    # Avoid division by zero if capacity - sums + 1 == 0?
    # sums is float. capacity is 150.
    # If sums = 149, waste = 1, denom = 2.
    # If sums = 150, waste = 0, denom = 1.
    # If sums = 151 (not fit), waste = -1, denom = 0. Division by zero!
    # We must handle the denominator carefully.
    
    # Let's compute the score only where fits is True to avoid issues, 
    # or use np.where with a safe denominator.
    
    # Safe denominator:
    # If fits, denom = (capacity - sums + 1.0)
    # If not fits, denom = 1.0 (so score is 0 or small, will be masked)
    
    waste = capacity - sums
    # Ensure denominator is positive and non-zero for calculation stability
    # For fitting pairs, waste >= 0, so waste + 1 >= 1.
    # For non-fitting pairs, waste < 0.
    # We want to avoid 0 or negative denominators in the raw calculation before masking?
    # Actually, np.where evaluates both branches.
    # So we must ensure the "false" branch of the denominator is safe.
    
    safe_waste_plus_one = np.where(fits, waste + 1.0, 1.0)
    denom = safe_waste_plus_one
    
    # Calculate raw scores
    raw_scores = products / denom
    
    # Apply the fit condition: non-fitting pairs get 1e-9
    H = np.where(fits, raw_scores, 1e-9)
    
    # Ensure no non-finite values (just in case)
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)
        
    # Ensure shape
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
        
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Return immediately if we are close to the limit
        return H

    return H
