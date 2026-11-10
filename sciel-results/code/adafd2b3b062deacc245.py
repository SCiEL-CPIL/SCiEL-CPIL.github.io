import numpy as np
import itertools
import time
import sys

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Precompute squared norms for grid points
    grid_sq_norms = (grid ** 2).sum(axis=1)
    
    def is_valid_addition(arr_indices, p_idx):
        """
        Checks if adding grid[p_idx] to the set defined by arr_indices violates the constraint.
        arr_indices: array of indices of current points.
        p_idx: index of candidate point.
        """
        m = len(arr_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq_norms[p_idx]
        
        # Points in current set
        pts = grid[arr_indices]
        pts_sq = grid_sq_norms[arr_indices]
        
        # We want to check if there exist 4 points in pts such that p is on their sphere.
        # Sphere equation: |x|^2 + d1*x + d2*y + d3*z + d4 = 0
        # For a specific sphere defined by 4 points, we can solve for d.
        # Instead of checking all C(m, 4) subsets, we can use a projection trick.
        
        # Let L be the (m, 4) matrix where row i is [x_i, y_i, z_i, |x_i|^2]
        # Let b be the (m,) vector |x_i|^2.
        # Wait, the equation is |x|^2 + D . x + E = 0.
        # So D . x + E = -|x|^2.
        
        # If we pick 4 points, we solve D . X_4 + E * 1 = -|X_4|^2.
        # Then we check if D . p + E = -|p|^2.
        
        # Vectorized approach:
        # We can't just do lstsq on the whole set because the set might not be "spherical" in a way that allows a single solution for all points.
        # However, if p is on a sphere defined by ANY 4 points, then those 4 points plus p are co-spherical.
        
        # The current program checks all C(m, 4). This is O(m^4).
        # Can we do better?
        
        # If m is large, C(m, 4) is huge.
        # But note: if p is on a sphere defined by 4 points, it means the rank of the matrix formed by the 5 points (lifted) is < 5.
        
        # Let's try to batch the determinant check more aggressively using NumPy's broadcasting if m is small, 
        # or use a smarter heuristic if m is large.
        
        # Actually, for m > 50, C(m, 4) is ~250,000. 
        # 250k determinants of 4x4 matrices.
        # np.linalg.det is slow in a loop.
        
        # Alternative: 
        # We can check if the rank of the (m+1) x 5 matrix (lifted points) is < 5? 
        # No, that checks if ALL points are on a sphere. We only need to check if p is on a sphere defined by 4 others.
        
        # Let's stick to the determinant check but optimize the inner loop.
        # We will generate combinations in batches and compute determinants using a vectorized formula for 4x4 determinants.
        
        def batch_det_4x4(matrices):
            # matrices: (N, 4, 4)
            # Returns: (N,) determinants
            N = matrices.shape[0]
            if N == 0:
                return np.array([])
            
            # Extract rows
            r1 = matrices[:, 0, :]
            r2 = matrices[:, 1, :]
            r3 = matrices[:, 2, :]
            r4 = matrices[:, 3, :]
            
            # det = r1 . (r2 x r3) ? No, that's 3x3.
            # For 4x4, we can use the formula:
            # det(M) = sum_{sigma} sign(sigma) prod M[i, sigma(i)]
            # There are 24 terms.
            
            # Let's use np.linalg.det in batches of 1000. It's BLAS backed.
            # If N is small, just do it.
            dets = np.empty(N, dtype=np.float64)
            batch_size = 1000
            for i in range(0, N, batch_size):
                batch = matrices[i:i+batch_size]
                dets[i:i+batch_size] = np.linalg.det(batch)
            return dets

        # Build the matrix W for all points in arr
        # Row: [q - p, |q|^2 - |p|^2]
        # This is (m, 4)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = pts - p
        w[:, 3] = pts_sq - p_sq
        
        # We need to check if any 4 rows have det < 0.5
        # If m is small (< 30), do it directly.
        if m < 30:
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True
        
        # For larger m, we can try to find violations faster.
        # Observation: If a violation exists, it might be found early in a random permutation of indices?
        # But we need determinism and exactness.
        
        # Let's process combinations in batches.
        # To avoid generating a massive list, we iterate.
        # However, iterating C(m, 4) in Python is slow.
        
        # Optimization: 
        # If we sort the indices, we can't easily batch.
        # Let's just do the loop but break early.
        # And use the batched det function.
        
        batch = []
        batch_limit = 1000
        for combo in itertools.combinations(range(m), 4):
            batch.append(combo)
            if len(batch) == batch_limit:
                indices = np.array(batch, dtype=np.int32)
                sub_mats = w[indices]
                dets = batch_det_4x4(sub_mats)
                if np.any(np.abs(dets) < 0.5):
                    return False
                batch = []
        if batch:
            indices = np.array(batch, dtype=np.int32)
            sub_mats = w[indices]
            dets = batch_det_4x4(sub_mats)
            if np.any(np.abs(dets) < 0.5):
                return False
                
        return True

    def solve():
        start_time = time.time()
        deadline = start_time + 580 # Safety margin
        
        best_points = np.empty((0, 3), dtype=np.int64)
        best_size = 0
        
        np.random.seed(42)
        
        restart_count = 0
        while time.time() < deadline:
            restart_count += 1
            
            # Phase 1: Greedy construction
            order = np.random.permutation(num_points)
            current_indices = []
            
            for p_idx in order:
                if is_valid_addition(np.array(current_indices), p_idx):
                    current_indices.append(p_idx)
                    
            # Phase 2: Local Search / Shake
            # Try to add points that were rejected, or swap points to allow new additions.
            # Simple shake: remove a random point from the current set, then try to add it back 
            # or add a different point that might now fit.
            
            # Actually, a better shake: 
            # 1. Remove k random points from the set.
            # 2. Run greedy on the remaining points + the removed points + any other unselected points.
            
            # Let's try a simpler improvement: 
            # Iterate through all unselected points. If adding one violates the constraint, 
            # can we remove one existing point to make it fit?
            # This is complex. Let's stick to multiple restarts with different random seeds/orders.
            
            # But we can do a "second pass" on the current set with a different order of unselected points.
            selected_set = set(current_indices)
            unselected = [i for i in range(num_points) if i not in selected_set]
            
            # Shuffle unselected
            np.random.shuffle(unselected)
            
            for p_idx in unselected:
                if is_valid_addition(np.array(current_indices), p_idx):
                    current_indices.append(p_idx)
                    selected_set.add(p_idx)
                    
            if len(current_indices) > best_size:
                best_size = len(current_indices)
                best_points = grid[np.array(current_indices)]
                
            if time.time() > deadline - 30:
                break
                
        return best_points

    return solve()

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
