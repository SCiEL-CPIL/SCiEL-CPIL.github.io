# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points as possible from the 8x8x8 integer grid {0,...,7}^3
    such that no five selected points are co-spherical (or co-planar).
    
    Strategy:
    1. Greedy randomized search with vectorized conflict detection.
    2. Conflict detection: Five points p, q1, q2, q3, q4 are co-spherical iff
       det of matrix with rows [qi - p, |qi|^2 - |p|^2] is 0.
       We batch these determinants for all 4-subsets of the current set.
    3. Multiple random restarts within a strict time budget.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    r2 = (grid ** 2).sum(axis=1)  # squared norms

    # Precompute combinations for small m to avoid overhead?
    # Actually, generating combinations is fast. The bottleneck is det.
    # We will generate on the fly.
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit (20s buffer)
    
    np.random.seed(42)
    best_len = 0
    best_set = []
    
    def can_add(p_idx, S_indices):
        """
        Check if point p_idx can be added to the set defined by S_indices.
        S_indices is a list of indices into grid.
        """
        m = len(S_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        pr2 = r2[p_idx]
        
        # Get the actual indices of the selected points
        # S_indices is a list of ints
        # We need all combinations of 4 from S_indices
        # If m is large, this is expensive. 
        # m=10 -> 210 combos. m=20 -> 4845 combos.
        
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return True
            
        K = len(combos)
        # Convert to numpy array for vectorized indexing
        # Shape (K, 4)
        combo_arr = np.array(combos, dtype=np.int32)
        
        # Index into the selected points
        # S_indices is a list, convert to array for indexing
        S_arr = np.array(S_indices, dtype=np.int32)
        
        # Get coordinates of the 4 points in each combo
        # S_arr[combo_arr] has shape (K, 4, 3)
        qs_c = grid[S_arr[combo_arr]]
        # Get squared norms
        qr2_c = r2[S_arr[combo_arr]]
        
        # Construct the 4x4 matrices
        # Row i is [qi - p, |qi|^2 - |p|^2]
        
        # diff: (K, 4, 3) - (3,) -> (K, 4, 3)
        diff = qs_c - p
        # diff_r2: (K, 4) - scalar -> (K, 4)
        diff_r2 = qr2_c - pr2
        
        # Stack into (K, 4, 4)
        mats = np.empty((K, 4, 4), dtype=np.float64)
        mats[:, :, :3] = diff
        mats[:, :, 3] = diff_r2
        
        # Compute determinants
        dets = np.linalg.det(mats)
        
        # Check for singularity (threshold 0.5 as per task)
        if np.any(np.abs(dets) < 0.5):
            return False
        return True

    # Main search loop
    # We run multiple restarts. Each restart is a greedy pass.
    # To maximize k, we just keep the best found.
    
    # We can also try to improve the order of points in the grid to bias the greedy choice?
    # Random permutation is a good baseline.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random permutation of grid indices
        order = np.random.permutation(N)
        
        S_indices = []
        
        for p_idx in order:
            # Check if we can add p_idx
            if can_add(p_idx, S_indices):
                S_indices.append(p_idx)
        
        if len(S_indices) > best_len:
            best_len = len(S_indices)
            best_set = S_indices.copy()
            
            # If we find a very good set, we might want to try local search?
            # But given the time, more restarts is usually better.
            # If best_len is high, we can reduce the time per restart?
            # No, the time per restart is dominated by the greedy pass.
            
            # Early termination if we hit a theoretical max? 
            # Max possible is 512. But no 5 co-spherical is a strong constraint.
            # Known bounds for 3D grid: 
            # For 8x8x8, max k is likely around 10-20.
            # If we reach 20, we can continue, but it's unlikely to go much higher.
            
    if not best_set:
        # Fallback: return a small valid set
        # First 4 points are always valid (no 5 points to be co-spherical)
        best_set = [0, 1, 2, 3]
        
    return grid[best_set].astype(np.int64)

# EVOLVE-BLOCK-END
