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
        gaps = capacity - sums  # shape (n, n), gaps[i,j] = capacity - d[i] - d[j]
        
        # For each valid pair (i,j), find the maximum size of a third item k
        # that fits in the remaining space (gaps[i,j]).
        # This is: max_k { d[k] : d[k] <= gaps[i,j] }
        # If no such k exists (or max is 0), the pair nearly closes the bin.
        
        # Sort demands for efficient lookup
        d_sorted = np.sort(d)
        
        # For each valid pair, compute the largest item that fits in the gap
        # Use searchsorted to find the rightmost position where d_sorted <= gap
        # gaps has shape (n, n), we need to do this for valid entries only
        
        # Vectorized approach: for each gap value, find max fitting item
        # searchsorted returns insertion point, so we subtract 1 to get the index
        # of the largest element <= gap
        gap_vals = gaps[valid]  # 1D array of gap values for valid pairs
        
        # Find the largest demand that fits in each gap
        # np.searchsorted(d_sorted, gap_vals, side='right') gives index after last <= gap
        idx = np.searchsorted(d_sorted, gap_vals, side='right')
        # The largest fitting item is at index idx-1 (if idx > 0)
        max_fitting = np.where(idx > 0, d_sorted[np.maximum(idx - 1, 0)], 0.0)
        
        # Bin-completion potential: reward pairs where the remaining space
        # cannot fit any other item (max_fitting == 0) or can only fit very small items
        # This directly rewards moves that close bins
        
        # Primary signal: inverse of (max_fitting + epsilon)
        # When max_fitting is 0 (no third item fits), this is very large
        # When max_fitting is large, this is small
        eps = 1e-6
        completion_score = 1.0 / (max_fitting + eps)
        
        # Secondary signal: inverse gap (strong best-fit preference)
        inv_gap = 1.0 / (gap_vals + eps)
        
        # Tertiary signal: fill ratio (higher fill is better)
        ratio = sums[valid] / capacity
        fill_score = ratio ** 2.0
        
        # Blend: combine all three signals
        # completion_score is the key new signal - it peaks when the pair closes the bin
        # We use a weighted product to balance the signals
        # Normalize each signal to [0, 1] range for blending
        # completion_score ranges from 1/eps (when max_fitting=0) to 1/(max_d+eps)
        # Scale completion_score: map to [0,1] where 1 means bin is closed
        comp_max = 1.0 / eps
        comp_min = 1.0 / (d_sorted[-1] + eps) if d_sorted[-1] > 0 else 1.0
        comp_norm = (completion_score - comp_min) / (comp_max - comp_min) if comp_max > comp_min else 0.5
        comp_norm = np.clip(comp_norm, 0.0, 1.0)
        
        # inv_gap ranges from 1/(capacity) to 1/eps
        inv_gap_max = 1.0 / eps
        inv_gap_min = 1.0 / capacity
        inv_gap_norm = (inv_gap - inv_gap_min) / (inv_gap_max - inv_gap_min) if inv_gap_max > inv_gap_min else 0.5
        inv_gap_norm = np.clip(inv_gap_norm, 0.0, 1.0)
        
        # Combine: emphasize bin-completion, with gap and fill as secondary
        eta_valid = (0.5 * comp_norm + 0.3 * inv_gap_norm + 0.2 * fill_score)
        
        # Ensure strictly positive
        eta_valid = np.maximum(eta_valid, 1e-9)
        
        eta[valid] = eta_valid
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
    
    return eta
