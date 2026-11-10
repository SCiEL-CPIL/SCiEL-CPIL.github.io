# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large set of points in the 8x8x8 grid such that no 5 points are 
    co-spherical (or co-planar).
    
    Strategy:
    1. Multiple randomized greedy passes.
    2. Local improvement using "2-step swaps" (remove one, add one, then try to add another).
    3. Deadline guard to ensure completion within 600s.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms for grid points
    sq_norms = (grid ** 2).sum(axis=1)

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Checks if point p can be added to set arr without creating 5 co-spherical points.
        arr: (k, 3) array of existing points.
        p: (3,) array of candidate point.
        """
        m = len(arr)
        if m < 4:
            return True
            
        # Construct the 4x4 matrix for each combination of 4 points?
        # Efficient check:
        # The condition is that for any 4 points q1..q4 in arr, 
        # det([q_i - p; |q_i|^2 - |p|^2]) != 0.
        
        # Vectorized approach for small m:
        # If m is large, this is expensive.
        # We can check the rank of the full matrix of lifted points including p.
        # But we only need to check subsets of size 4 from arr + p?
        # No, if any 4 from arr are dependent with p, it's bad.
        # This is equivalent to: The rank of the matrix formed by 
        # [q_i - p, |q_i|^2 - |p|^2] for i in 1..m must be at least 4?
        # No. If the rank is < 4, then there exist 4 linearly dependent rows?
        # Not exactly. If the rank is < 4, the 4x4 determinant of some subset is 0.
        # So we just need to check if the rank of the (m x 4) matrix is < 4?
        # Wait. If the rank is 3, then ALL 4x4 subdeterminants are 0.
        # If the rank is 4, it's possible that SOME 4x4 subdeterminants are 0 
        # and others are not.
        # The condition is "NO 5 points are co-spherical".
        # This means for EVERY subset of 4 points from arr, the determinant with p must be non-zero.
        # So we need to check ALL C(m, 4) determinants.
        
        # Optimization: 
        # If m is small, we can iterate.
        # If m is large, we can use the property that if the points are in general position,
        # most determinants are non-zero.
        
        # Let's use a hybrid:
        # For m < 30, compute all determinants.
        # For m >= 30, we can sample or use a faster approximation? 
        # Actually, for m up to ~100, C(100,4) is ~4M. 
        # Computing 4M 4x4 determinants in Python loop is slow.
        # In NumPy, we can stack them and use np.linalg.det.
        
        # Construct all combinations
        # This is memory intensive for large m.
        # Let's stick to a loop but optimize the inner determinant calculation?
        # Or use the fact that we only need to check if ANY is zero.
        
        # Better approach for speed:
        # Check if the rank of the matrix W (m x 4) is 4.
        # If rank < 4, return False immediately.
        # If rank == 4, we still need to ensure no subset of 4 is dependent.
        # However, if the rank is 4, it is possible that a specific subset of 4 is dependent 
        # (det=0) while the overall rank is 4.
        # Example: Rows r1, r2, r3, r4, r5. 
        # If r1,r2,r3,r4 are dependent, det=0.
        # But r1,r2,r3,r5 might be independent.
        # So rank(W) == 4 does NOT guarantee validity.
        
        # We must check all subsets.
        
        # To speed up, we can break early.
        
        # Pre-compute differences and norm diffs
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = sq_norms[arr_indices] - (p ** 2).sum() # Need indices
        
        # This function takes arr, which is a subset of grid.
        # We need to map arr back to sq_norms.
        # It's easier to pass indices or precompute everything.
        
        # Let's refactor to use indices for efficiency.
        pass

    # Refactored for speed using indices
    def can_add_idx(arr_idx: np.ndarray, p_idx: int) -> bool:
        """
        arr_idx: indices of existing points in grid.
        p_idx: index of candidate point in grid.
        """
        m = len(arr_idx)
        if m < 4:
            return True
            
        p = grid[p_idx]
        arr = grid[arr_idx]
        
        # Calculate the 4th component: |q|^2 - |p|^2
        # sq_norms is precomputed for all grid points
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = sq_norms[arr_idx] - sq_norms[p_idx]
        
        # Check all combinations of 4
        # For performance, if m is large, this is the bottleneck.
        # We can use a random sampling strategy if m is very large?
        # No, we need strict validity.
        
        # Optimization: 
        # If we find one det < 0.5, return False.
        
        # Use itertools.combinations for indices
        # This is slow in Python.
        # Let's try to vectorize the determinant calculation for batches.
        
        # For now, let's use a loop but with early exit.
        # And we will limit the number of restarts based on time.
        
        # Actually, for m up to 100, C(100,4) is ~4M.
        # A Python loop of 4M iterations is too slow (~10s per check).
        # We need a faster check.
        
        # Alternative: 
        # The condition is equivalent to:
        # The matrix M of size (m+1) x 5 with rows [x, y, z, x^2+y^2+z^2, 1] 
        # must have the property that any 5 rows are linearly independent?
        # No, "no 5 points are co-spherical" means any 5 points are NOT co-spherical.
        # So any 5 rows of M must be linearly independent?
        # If 5 points are co-spherical, the 5x5 determinant is 0.
        # So yes, any 5x5 submatrix of M must have non-zero determinant.
        
        # This is the same as checking all C(m+1, 5) determinants of M.
        # This is even more expensive.
        
        # Let's go back to the 4x4 check relative to p.
        # We can speed up the combination generation and det calculation using NumPy.
        
        # Generate all combinations of 4 indices
        if m < 50:
            # Small m, direct check
            indices = np.arange(m)
            # Use itertools for simplicity, but vectorize the det part?
            # No, itertools yields tuples.
            
            # Let's use a different approach:
            # Compute the rank of W. If rank < 4, False.
            # If rank == 4, we still need to check subsets.
            
            # Actually, for random points, the probability of a specific 4x4 det being 0 is 0.
            # The only way it's 0 is if the points are structurally co-spherical.
            # We can check a few random subsets? No, must be exact.
            
            # Let's use the fact that we can compute the determinant of the full m x 4 matrix?
            # No.
            
            # Let's try to optimize the Python loop by using a C-extension-like logic?
            # We can't.
            
            # Let's reduce the problem size.
            # If we use a greedy approach, m grows.
            # Maybe we can use a "sliding window" or "local" check?
            # No, the constraint is global.
            
            # Let's assume m is small enough for the loop to be fast if we vectorize the det.
            # We can stack the 4x4 matrices and call np.linalg.det once.
            
            combos = list(itertools.combinations(range(m), 4))
            if not combos:
                return True
            
            # Build a 3D array of shape (C, 4, 4)
            # This is memory heavy for large C.
            # C(100, 4) = 3,921,225.
            # 3.9M * 16 bytes = 62 MB. Acceptable.
            # C(200, 4) = 64M. 1 GB. Too much.
            
            # So for m > 100, we need a different strategy.
            # But greedy usually doesn't go beyond 100-150 for this problem.
            
            # Let's implement the vectorized det check.
            sub = w[np.array(combos)]
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)

    # Improved can_add for larger m using random sampling + exact check on samples?
    # No, must be exact.
    
    # Let's stick to the vectorized version for m < 120.
    # For m >= 120, we might timeout.
    # We will add a deadline guard.

    start_time = time.time()
    deadline = 550.0  # Safety margin
    
    best_set = np.array([], dtype=np.int64)
    
    def greedy_pass(order: np.ndarray) -> np.ndarray:
        current_set = []
        for idx in order:
            if time.time() > deadline:
                break
            if can_add_idx(np.array(current_set), idx):
                current_set.append(idx)
        return np.array(current_set, dtype=np.int64)

    def local_improve(set_idx: np.ndarray) -> np.ndarray:
        """
        Try to improve the set size by 2-step swaps.
        Remove u, add v, then add w.
        """
        set_idx = list(set_idx)
        set_set = set(set_idx)
        all_indices = set(range(N))
        not_in_set = all_indices - set_set
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all pairs (u, v)
            # To save time, we can limit the number of attempts per iteration
            # or shuffle the pairs.
            
            # Create list of (u, v) pairs
            # This is O(N^2) which is 262k pairs.
            # For each pair, we check validity of (set - u + v) and then try to add w.
            # This is too slow if done for all pairs every iteration.
            
            # Strategy: 
            # 1. Pick a random u from set.
            # 2. Pick a random v from not_in_set.
            # 3. Check if set - u + v is valid.
            # 4. If valid, try to add any w from not_in_set (v != w).
            # 5. If w can be added, commit.
            
            # We will do a fixed number of random attempts per "round".
            
            for _ in range(1000): # 1000 attempts per round
                if time.time() > deadline:
                    break
                
                if not set_idx:
                    break
                
                u = np.random.choice(set_idx)
                v = np.random.choice(list(not_in_set))
                
                # Remove u, add v
                new_set = [x for x in set_idx if x != u]
                new_set.append(v)
                
                # Check validity of new_set
                # We need to check if v can be added to (set - u)
                # This is the same as can_add_idx(set - u, v)
                # But we also need to ensure the rest of the set is still valid?
                # Yes, removing a point doesn't invalidate the rest.
                # So we only need to check if v is compatible with (set - u).
                
                # Optimization: 
                # can_add_idx expects an array.
                # We can check if v is compatible with the existing points.
                
                if can_add_idx(np.array(new_set[:-1]), v):
                    # new_set is valid so far
                    # Now try to add a third point w
                    # We need to find any w in (not_in_set - {v}) that can be added to new_set
                    
                    # To save time, we try a few random w's
                    candidates_w = [x for x in not_in_set if x != v]
                    if candidates_w:
                        # Shuffle and try a few
                        np.random.shuffle(candidates_w)
                        for w in candidates_w[:10]: # Try top 10 random
                            if can_add_idx(np.array(new_set), w):
                                # Success!
                                new_set.append(w)
                                set_idx = new_set
                                set_set = set(set_idx)
                                not_in_set = all_indices - set_set
                                improved = True
                                break
            if improved:
                # Reset the "improved" flag to continue searching?
                # Or just break and restart?
                # Let's continue the while loop to find more improvements.
                pass
                
        return np.array(set_idx, dtype=np.int64)

    np.random.seed(0)
    
    # Run multiple greedy passes
    num_restarts = 50 # We will break early if time runs out
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(N)
        greedy_result = greedy_pass(order)
        
        if len(greedy_result) > len(best_set):
            best_set = greedy_result
        
        # Apply local improvement
        if time.time() < deadline - 10: # Leave some time for final check
            improved_set = local_improve(best_set)
            if len(improved_set) > len(best_set):
                best_set = improved_set

    return best_set

# EVOLVE-BLOCK-END
