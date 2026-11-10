# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points from the 8x8x8 grid {0,...,7}^3 such that no five
    are co-spherical.
    """
    n = 8
    deadline = time.time() + 590
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute features for all grid points: (|p|^2, x, y, z, 1)
    grid_f = grid.astype(np.float64)
    sq_norms = (grid_f ** 2).sum(axis=1)
    features = np.column_stack((sq_norms, grid_f, np.ones(N)))
    
    # features shape: (512, 5)
    
    def solve_null_vector(M):
        """
        M is 4x5. Find null vector v such that M @ v = 0.
        Returns normalized v.
        """
        # Use SVD
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        # The last row of Vh corresponds to the smallest singular value
        v = Vh[-1, :]
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return None
        return v / norm

    def can_add_fast(constraints, f_p):
        """
        constraints: (N_c, 5) array
        f_p: (5,) array
        Returns True if no constraint is violated.
        """
        if len(constraints) == 0:
            return True
        vals = constraints @ f_p
        return not np.any(np.abs(vals) < 1e-6)

    def greedy_with_order(order):
        points = []
        constraints = np.empty((0, 5), dtype=np.float64)
        
        for idx in order:
            p = grid[idx]
            f_p = features[idx]
            
            if can_add_fast(constraints, f_p):
                points.append(idx)
                
                # Generate new constraints from p and all triples in points[:-1]
                if len(points) >= 4:
                    # points[:-1] are the previous points
                    prev_indices = points[:-1]
                    m = len(prev_indices)
                    if m >= 3:
                        new_constraints = []
                        # Iterate over all triples in prev_indices
                        # To speed up, we can vectorize the SVD? 
                        # SVD is hard to vectorize easily. 
                        # But m is small initially.
                        
                        # Optimization: Only add constraints if the set is small?
                        # Or just do it.
                        
                        # We need to form M for each triple + p
                        # M rows: features of the 4 points.
                        
                        # Let's get the features of prev_indices
                        prev_feats = features[prev_indices] # (m, 5)
                        f_new = features[idx]
                        
                        # Generate combinations of 3 from m
                        # If m is large, this is slow.
                        # But we only do this when adding a point.
                        
                        # For m > 100, this might be slow.
                        # We can limit the number of constraints?
                        # No, we need all to be valid.
                        
                        # Let's try to vectorize the construction of M?
                        # We can't easily SVD a batch.
                        
                        # Alternative: Check if we can skip some constraints?
                        # No.
                        
                        # Let's just loop.
                        # To speed up, we can precompute the combinations of indices?
                        # No, m changes.
                        
                        # For now, simple loop.
                        # If m is large, this becomes the bottleneck.
                        # We can stop adding new constraints if the number of constraints is too high?
                        # No, that breaks validity.
                        
                        # Let's hope k stays under 50 where m^3 is small.
                        # Or we can use a different strategy for large k.
                        
                        # For now, proceed.
                        for c in itertools.combinations(range(m), 3):
                            i, j, k_idx = c
                            M = np.vstack([prev_feats[i], prev_feats[j], prev_feats[k_idx], f_new])
                            v = solve_null_vector(M)
                            if v is not None:
                                new_constraints.append(v)
                        
                        if new_constraints:
                            new_c_array = np.array(new_constraints)
                            constraints = np.vstack([constraints, new_c_array])
                            
        return np.array(points, dtype=np.int64)

    # Precompute orderings
    center = np.array([3.5, 3.5, 3.5])
    dists_from_center = np.linalg.norm(grid - center, axis=1)
    manhattan = np.abs(grid - 3.5).sum(axis=1)
    
    orderings = []
    orderings.append(np.argsort(dists_from_center))
    orderings.append(np.argsort(-dists_from_center))
    orderings.append(np.argsort(manhattan))
    orderings.append(np.argsort(-manhattan))
    
    for seed in [42, 123, 777, 999, 2024, 3141, 2718, 1618, 1414, 1732]:
        rng = np.random.RandomState(seed)
        orderings.append(rng.permutation(N))
        
    best = np.empty((0, 3), dtype=np.int64)
    best_len = 0
    
    # Phase 1: Greedy
    for order in orderings:
        if time.time() > deadline - 30:
            break
        # We need to return points, not indices.
        # greedy_with_order returns indices.
        indices = greedy_with_order(order)
        if len(indices) > best_len:
            best_len = len(indices)
            best = grid[indices]
            
    # Phase 2: More random restarts
    rng2 = np.random.RandomState(555)
    for i in range(200):
        if time.time() > deadline - 30:
            break
        perm = rng2.permutation(N)
        indices = greedy_with_order(perm)
        if len(indices) > best_len:
            best_len = len(indices)
            best = grid[indices]
            
    return best

# EVOLVE-BLOCK-END
