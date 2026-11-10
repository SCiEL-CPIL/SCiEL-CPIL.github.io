# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Find the maximum number of points in an 8x8x8 grid such that no 5 are co-spherical.
    Uses randomized greedy with multiple restarts and a time limit.
    """
    n = 8
    # Precompute grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    # Precompute squared norms for faster distance calculations
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)

    def can_add(arr_idx: np.ndarray, p_idx: int) -> bool:
        m = len(arr_idx)
        if m < 4:
            return True
        
        # Get coordinates of existing points
        arr = grid[arr_idx]
        p = grid[p_idx]
        
        # Construct the 4x4 matrix for each 4-combination of existing points
        # Rows: (q - p, |q|^2 - |p|^2)
        # We need to check if any 4 rows are linearly dependent.
        # This is equivalent to checking if the 5 points (p + 4 existing) are co-spherical.
        
        # Vectorized construction of the matrix W (m x 4)
        # W[:, 0:3] = arr - p
        # W[:, 3] = sq_norms[arr_idx] - sq_norms[p_idx]
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = sq_norms[arr_idx] - sq_norms[p_idx]
        
        # We need to check all 4-combinations of the m rows.
        # If m is small, we can use itertools.
        # If m is large, this is expensive.
        # Optimization: If m > 20, we can sample? No, must be exact.
        
        # Generate indices for combinations
        # For m=100, C(100,4) ~ 4M. 
        # Creating a 4M x 4 x 4 array is 64M floats ~ 512MB. This might be too much or slow.
        # Let's check memory. 4M * 4 * 4 * 8 bytes = 512 MB. 
        # This is likely to cause memory issues or be slow in numpy.
        
        # Alternative: Loop over the first index i, then vectorize the rest?
        # Or just loop in Python? Python loops over 4M items are slow.
        
        # Let's try to do it in chunks or use a more memory-efficient approach.
        # Actually, np.linalg.det on a large array is fast, but the array creation is the killer.
        
        # Let's try a different approach: 
        # Iterate over combinations of 3 indices (i, j, k) and check if the 4th point (l) 
        # makes the determinant 0? No, we need to check if ANY 4 points with p are dependent.
        
        # Let's stick to the vectorized approach but be careful with memory.
        # If m is large, we might need to break it down.
        
        # For now, let's assume m stays small enough or we optimize the combination generation.
        # Using itertools.combinations returns tuples. 
        # np.array(list(...)) is the bottleneck.
        
        # Let's try to use np.indices? No.
        
        # We will use the vectorized approach but add a safety check for memory/time.
        # If m > 30, C(30,4) = 27405. 27405 * 4 * 4 * 8 = 3.5 MB. Safe.
        # If m > 50, C(50,4) = 230300. 230300 * 4 * 4 * 8 = 29 MB. Safe.
        # If m > 80, C(80,4) = 1.5M. 1.5M * 4 * 4 * 8 = 192 MB. Risky.
        # If m > 100, C(100,4) = 3.9M. 3.9M * 4 * 4 * 8 = 500 MB. Very Risky.
        
        # We should avoid creating the full array if m is large.
        # Instead, we can loop over the first index i, and for each i, 
        # form the matrix for combinations of the remaining m-1 points taken 3 at a time?
        # No, the determinant is of the 4x4 matrix of the 4 points.
        
        # Let's use a loop over the first index i, and vectorize the check for the other 3?
        # Actually, the determinant of the 4x4 matrix formed by rows w[i], w[j], w[k], w[l] is 0.
        # We can fix i, and check all (j,k,l) from the remaining points.
        # This reduces the problem to checking C(m-1, 3) determinants for each i.
        # Total checks is still C(m, 4).
        # But we can create a (m-1) x 4 array for the remaining points, 
        # and then we need to check if any 3 of them with w[i] are dependent.
        # This is equivalent to checking if w[i] lies in the span of any 3 others? No.
        
        # Let's just use the full vectorized approach for m < 60, and a slower loop for m >= 60.
        # Or, we can just let it run and hope m doesn't get too high.
        # Given the problem, m is likely < 100.
        
        if m < 60:
            combos = np.array(list(itertools.combinations(range(m), 4)))
            sub = w[combos]
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Fallback: Python loop (slow but memory safe)
            # We can optimize this by pre-computing something?
            # For now, just loop.
            for c in itertools.combinations(range(m), 4):
                sub = w[list(c)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True

    start_time = time.time()
    timeout = 550  # Safety margin
    
    best_arr = np.empty((0, 3), dtype=np.int64)
    best_len = 0
    
    rng = np.random.default_rng(0)
    
    while time.time() - start_time < timeout:
        order = rng.permutation(N)
        arr_idx = []
        
        for p_idx in order:
            if can_add(np.array(arr_idx), p_idx):
                arr_idx.append(p_idx)
                
        if len(arr_idx) > best_len:
            best_len = len(arr_idx)
            best_arr = grid[arr_idx]
            
    return best_arr.astype(np.int64)


# EVOLVE-BLOCK-END
