import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    # Precompute squared norms
    norms = (grid ** 2).sum(axis=1)
    
    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        
        # Lifted vectors relative to p
        # w_i = (q_i - p, |q_i|^2 - |p|^2)
        # We check if any 4 rows of W are linearly dependent.
        # This is equivalent to checking if the rank of W is < 4? No.
        # We need to check if any 4x4 submatrix has det 0.
        
        # Optimization: If m is small, just check all.
        # If m is large, this is slow.
        # But let's assume m is manageable.
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = norms[arr] - norms[p] # Wait, norms is indexed by grid index, not values.
        # We need to map arr to grid indices or precompute norms for all possible points.
        # Better: Precompute a dictionary or 3D array for norms.
        # Or just compute norms[p] and norms[arr] on the fly?
        # norms is precomputed for grid. arr contains points.
        # We can compute norms of arr directly.
        
        # Let's re-calculate norms for arr and p to be safe and simple, 
        # or pass a lookup.
        # Since grid is small (512), we can create a lookup table.
        
        pass

    # Lookup table for norms
    norm_lookup = np.zeros((n, n, n), dtype=np.int64)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                norm_lookup[x, y, z] = x*x + y*y + z*z

    def get_norms(points):
        return norm_lookup[points[:, 0], points[:, 1], points[:, 2]]

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = get_norms(arr) - get_norms(p)
        
        # Check all combinations of 4
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = w[combos]
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    # Greedy
    np.random.seed(0)
    best = None
    start_time = time.time()
    deadline = start_time + 500 # 500s limit
    
    for _ in range(100): # Many restarts
        if time.time() > deadline:
            break
        order = grid[np.random.permutation(len(grid))]
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if can_add(arr, p):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
        
        # Local Search
        # ... (implementation of local search)
        
        if best is None or len(arr) > len(best):
            best = arr

    return best
