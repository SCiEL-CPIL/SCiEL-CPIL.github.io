import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    # Precompute all grid points
    xs = np.arange(N)
    ys = np.arange(N)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    all_points = np.column_stack([XX.ravel(), YY.ravel()])  # shape (4096, 2)
    
    best_set = []
    best_size = 0
    
    # Precompute squared distances between all pairs of points?
    # 4096 x 4096 matrix of int64 is 16MB. Feasible.
    # But we only need distances from a candidate point to the current set.
    # Let's stick to dynamic calculation but vectorized.
    
    def check_and_add(p, points_arr, forbidden_mask):
        """
        Check if point p can be added.
        points_arr: (k, 2) array of current points.
        forbidden_mask: (k, max_dist) boolean array. 
                        forbidden_mask[i, d] is True if distance d is forbidden for point i.
                        Note: max_dist is (N-1)^2 + (N-1)^2 = 2*(N-1)^2.
        Returns (valid, new_points_arr, new_forbidden_mask) or (False, None, None).
        """
        k = len(points_arr)
        max_d = 2 * (N - 1) ** 2
        
        if k == 0:
            new_points = np.array([[p[0], p[1]]], dtype=np.int32)
            new_mask = np.zeros((1, max_d + 1), dtype=bool)
            return True, new_points, new_mask
        
        # Compute distances from p to all existing points
        dx = points_arr[:, 0].astype(np.int64) - p[0]
        dy = points_arr[:, 1].astype(np.int64) - p[1]
        dists = dx * dx + dy * dy  # shape (k,)
        
        # Check 1: All distances from p must be distinct
        # If any distance repeats, it forms an isosceles triangle with p as apex.
        if len(np.unique(dists)) != k:
            return False, None, None
        
        # Check 2: For each existing point i, the distance d_i must not be already forbidden.
        # forbidden_mask[i, d_i] should be False.
        # We can do this with advanced indexing.
        # indices: [0, 1, ..., k-1] for rows, dists for columns
        if forbidden_mask[np.arange(k), dists].any():
            return False, None, None
        
        # Build new arrays
        new_points = np.vstack([points_arr, [[p[0], p[1]]]])
        
        # New mask for the new point (index k)
        # The new point forbids its own distances to all previous points.
        new_row = np.zeros(max_d + 1, dtype=bool)
        new_row[dists] = True
        
        # Update existing rows: they now also forbid the distance to the new point.
        # forbidden_mask[i, dists[i]] = True
        # Note: dists[i] is the distance from point i to p.
        # This is symmetric.
        
        # We need to copy the old mask and update.
        # To avoid copying the whole large matrix every time (which is slow), 
        # we can use a list of sets? No, numpy is faster for bulk ops if we manage memory.
        # Actually, copying a (k, max_d) bool matrix is k * max_d bytes.
        # k ~ 20, max_d ~ 8000. 20*8000 = 160KB. Copying 160KB is fast.
        
        new_mask = np.zeros((k + 1, max_d + 1), dtype=bool)
        new_mask[:k, :] = forbidden_mask
        # Update existing rows
        new_mask[np.arange(k), dists] = True
        # Set new row
        new_mask[k, :] = new_row
        
        return True, new_points, new_mask

    def greedy_from_order(order, points_arr, forbidden_mask, time_limit):
        """Try to add points from order to the set."""
        for idx in order:
            if time.time() > time_limit:
                break
            p = all_points[idx]
            valid, new_pts, new_mask = check_and_add(p, points_arr, forbidden_mask)
            if valid:
                points_arr = new_pts
                forbidden_mask = new_mask
        return points_arr, forbidden_mask

    def local_search(points_arr, forbidden_mask, time_limit):
        """
        Try to improve by removing a point and trying to add two others.
        This is expensive, so we limit iterations.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(points_arr)
            if k == 0:
                break
            
            # Create a list of indices to try removing
            # We try removing a random subset or iterate through all
            # Let's iterate through all to be thorough if k is small, else random sample
            indices_to_try = list(range(k))
            random.shuffle(indices_to_try)
            
            for remove_idx in indices_to_try:
                if time.time() > time_limit:
                    break
                
                # Remove point at remove_idx
                # We need to rebuild the mask for the remaining points.
                # Since the mask depends on the set, we have to reconstruct or carefully update.
                # Reconstructing from scratch is O(k^2). k is small (~20-30).
                
                remaining_indices = [i for i in range(k) if i != remove_idx]
                rem_points = points_arr[remaining_indices]
                
                # Rebuild mask for rem_points
                rem_k = len(rem_points)
                max_d = 2 * (N - 1) ** 2
                rem_mask = np.zeros((rem_k, max_d + 1), dtype=bool)
                
                # Compute all pair distances for rem_points
                # Use broadcasting
                dx = rem_points[:, 0:1].astype(np.int64) - rem_points[1:, 0].astype(np.int64) # (k-1, k)
                dy = rem_points[:, 1:2].astype(np.int64) - rem_points[1:, 1].astype(np.int64)
                d_mat = dx * dx + dy * dy # (k-1, k)
                
                # Fill mask
                for i in range(rem_k - 1):
                    # d_mat[i, :] are distances from i to i+1 ... k-1
                    # d_mat[:, i] are distances from 0 ... i-1 to i+1 (shifted?)
                    # Let's just do it simply for correctness, k is small.
                    pass
                
                # Simpler rebuild:
                rem_mask = np.zeros((rem_k, max_d + 1), dtype=bool)
                for i in range(rem_k):
                    for j in range(i + 1, rem_k):
                        d = (rem_points[i, 0] - rem_points[j, 0])**2 + (rem_points[i, 1] - rem_points[j, 1])**2
                        rem_mask[i, d] = True
                        rem_mask[j, d] = True
                
                # Now try to add points
                # Candidates: all points not in rem_points
                in_set = np.zeros(N*N, dtype=bool)
                for pt in rem_points:
                    in_set[pt[0] * N + pt[1]] = True
                candidates = np.where(~in_set)[0]
                random.shuffle(candidates)
                
                added_count = 0
                new_rem_points = rem_points
                new_rem_mask = rem_mask
                
                for c_idx in candidates:
                    if added_count >= 2:
                        break
                    p = all_points[c_idx]
                    valid, new_pts, new_m = check_and_add(p, new_rem_points, new_rem_mask)
                    if valid:
                        new_rem_points = new_pts
                        new_rem_mask = new_m
                        added_count += 1
                
                # If we added more than we removed (1), we improved
                if added_count > 1:
                    points_arr = new_rem_points
                    forbidden_mask = new_rem_mask
                    improved = True
                    break
                elif added_count == 1:
                    # Same size, might help future iterations? 
                    # Usually we want to strictly increase size.
                    # Let's accept same size with different composition to escape local minima?
                    # It's risky. Let's stick to strict improvement for now to be safe.
                    pass
        
        return points_arr, forbidden_mask

    num_restarts = 10000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        points_arr = np.zeros((0, 2), dtype=np.int32)
        max_d = 2 * (N - 1) ** 2
        forbidden_mask = np.zeros((0, max_d + 1), dtype=bool)
        
        # Strategy: mix of random and structured initializations
        if restart % 5 == 0:
            # Random order
            order = np.random.permutation(N*N)
        elif restart % 5 == 1:
            # Row-major order
            order = np.arange(N*N)
        elif restart % 5 == 2:
            # Column-major order
            order = np.arange(N*N).reshape(N, N).T.ravel()
        elif restart % 5 == 3:
            # Diagonal order
            order = []
            for d in range(2*N - 1):
                for x in range(max(0, d-N+1), min(N, d+1)):
                    y = d - x
                    if 0 <= y < N:
                        order.append(x * N + y)
            order = np.array(order)
        else:
            # Random with bias toward spread (e.g., shuffle blocks)
            order = np.random.permutation(N*N)
        
        points_arr, forbidden_mask = greedy_from_order(order, points_arr, forbidden_mask, deadline)
        
        # Second pass with different order to fill gaps
        if time.time() < deadline - 5:
            order2 = np.random.permutation(N*N)
            points_arr, forbidden_mask = greedy_from_order(order2, points_arr, forbidden_mask, deadline)
        
        # Local search
        if time.time() < deadline - 10 and len(points_arr) >= 5:
            points_arr, forbidden_mask = local_search(points_arr, forbidden_mask, deadline - 5)
        
        if len(points_arr) > best_size:
            best_size = len(points_arr)
            best_set = [(int(pt[0]), int(pt[1])) for pt in points_arr]
            
            # Optional: If we found a very good set, do more local search on it?
            # The loop above does local search on every restart that gets there.
            # But maybe we should keep the best set and polish it more?
            # The current logic restarts from scratch. 
            # Let's add a "polish best" phase at the end if time permits.
    
    # Final polish on the best set found
    if best_size > 0 and time.time() < deadline - 5:
        # Reconstruct points_arr and mask from best_set
        best_arr = np.array(best_set, dtype=np.int32)
        k = len(best_arr)
        max_d = 2 * (N - 1) ** 2
        best_mask = np.zeros((k, max_d + 1), dtype=bool)
        for i in range(k):
            for j in range(i + 1, k):
                d = (best_arr[i, 0] - best_arr[j, 0])**2 + (best_arr[i, 1] - best_arr[j, 1])**2
                best_mask[i, d] = True
                best_mask[j, d] = True
        
        # Run local search on the best set found
        polished_arr, polished_mask = local_search(best_arr, best_mask, deadline)
        
        if len(polished_arr) > best_size:
            best_size = len(polished_arr)
            best_set = [(int(pt[0]), int(pt[1])) for pt in polished_arr]

    return best_set
