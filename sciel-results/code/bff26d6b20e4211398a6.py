import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands, dtype=float)
    
    # Compute residual space for all pairs
    # R[i, j] = capacity - demands[i] - demands[j]
    R = capacity - (demands[:, None] + demands[None, :])
    
    # Initialize heuristic matrix with a small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    # Find valid pairs where R >= 0
    valid_mask = R >= 0
    
    if np.any(valid_mask):
        # Sort demands to use searchsorted for counting
        sorted_demands = np.sort(demands)
        
        # For each valid pair, count how many items in the full set fit in the residual R
        # searchsorted returns the index of the first element > R (if side='left') 
        # or first element >= R (if side='right'? No, side='right' is first > R? 
        # Let's check: np.searchsorted([1,2,3], 2, side='right') -> 2 (index of 3). 
        # So it returns the count of elements <= R.
        # Wait, if R=2, elements <= 2 are 1, 2. Count is 2.
        # searchsorted([1,2,3], 2, side='right') returns 2. Correct.
        
        # We only need to calculate for valid pairs to save time, but 
        # searchsorted on the whole R matrix is fine if we mask afterwards.
        
        # Calculate counts for all R, then mask
        # R can be negative, searchsorted handles it (returns 0 if R < min(D))
        counts = np.searchsorted(sorted_demands, R, side='right')
        
        # Update H for valid pairs
        # Heuristic: (Sum) / (Count + 1)
        # This rewards larger sums and smaller counts (smaller gaps)
        
        # Avoid division by zero or very large numbers?
        # counts is between 0 and n.
        # Sum is between 0 and capacity.
        
        H_valid = (demands[:, None] + demands[None, :]) / (counts + 1)
        
        # Apply mask
        H[valid_mask] = H_valid[valid_mask]
        
        # Ensure no NaNs or Infs
        H = np.nan_to_num(H, nan=1e-9, posinf=1e-9, neginf=1e-9)
        
    return H
