"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Generate grid points
    x = np.arange(N)
    y = np.arange(N)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    num_points = len(points)
    
    # Precompute squared distances between all points? 
    # 4096x4096 is ~16M floats. 128MB. Acceptable.
    # But we can compute on the fly to save memory if needed. 
    # Let's compute full distance matrix for speed in local search.
    # d[i, j] = squared distance between points[i] and points[j]
    
    # Vectorized distance calculation
    # points shape: (M, 2)
    # dist_sq = (p1 - p2)^2
    # We can use broadcasting: (points[:, np.newaxis, :] - points[np.newaxis, :, :])
    # This creates a (M, M, 2) array. 4096*4096*2*8 bytes = ~256MB. 
    # This might be too much memory or slow to allocate.
    # Let's compute row by row or use a smaller block.
    # Actually, 256MB is usually okay in modern environments, but let's be safe.
    # We'll compute distances on the fly for the greedy phase, 
    # and maybe precompute for local search if time permits.
    
    # For the greedy phase, we need distances between 'available' and 'selected'.
    
    def get_dist_sq(p1, p2):
        """p1: (N, 2), p2: (M, 2). Returns (N, M) squared distances."""
        dx = p1[:, 0][:, np.newaxis] - p2[0][np.newaxis, :]
        dy = p1[:, 1][:, np.newaxis] - p2[1][np.newaxis, :]
        return dx**2 + dy**2

    # Better vectorized dist:
    def compute_dists(A, B):
        """A: (N,2), B: (M,2). Returns (N,M) squared distances."""
        # A^2 + B^2 - 2AB
        A_sq = np.sum(A**2, axis=1)[:, np.newaxis]
        B_sq = np.sum(B**2, axis=1)[np.newaxis, :]
        AB = A @ B.T
        return A_sq + B_sq - 2 * AB

    selected_indices = []
    selected_points = np.empty((0, 2))
    
    available_indices = np.arange(num_points)
    
    # Precompute squared distances from each point to all others? No, too big.
    # We will compute dist(available, selected) at each step.
    
    # To improve speed, we can maintain a list of selected points.
    
    # Heuristic: Max-Min Distance (Dispersion)
    # At each step, pick the valid point that is furthest from its nearest neighbor in 'selected'.
    # This helps spread points and reduces the chance of forming isosceles triangles 
    # with small bases.
    
    for _ in range(10): # 10 restarts with different random initializations? 
        # Actually, let's just do one good run with a specific ordering, 
        # then local search.
        
        # Randomize start to avoid bias? 
        # The grid is symmetric. 
        # Let's try a deterministic "spiral" or "grid" order first, 
        # then local search.
        
        pass

    # Let's implement a single robust run with vectorized checks.
    
    selected_indices = []
    selected_points = np.empty((0, 2))
    available_indices = np.arange(num_points)
    
    # Cache for pairwise distances in selected set?
    # We need to check if dist(p, a) == dist(a, b) for a, b in selected.
    # This requires knowing all pairwise distances in selected.
    # selected_size K. K^2 distances.
    
    for step in range(num_points):
        if len(available_indices) == 0:
            break
            
        # Get points for available and selected
        avail_pts = points[available_indices]
        
        if len(selected_indices) == 0:
            # First point: pick any, e.g., (0,0) or center?
            # Center might be better for dispersion? 
            # Let's pick (0,0) for determinism, or random.
            # Let's pick the center of the grid to maximize spread potential.
            center_idx = (N//2) * N + (N//2)
            # But center might be highly symmetric. 
            # Let's just pick index 0 for now, local search will fix.
            # Actually, picking a corner is often better for packing.
            p_idx = available_indices[0]
            selected_indices.append(p_idx)
            selected_points = np.vstack([selected_points, points[p_idx]])
            available_indices = np.delete(available_indices, 0)
            continue
        
        sel_pts = points[selected_indices]
        
        # Compute distances from all available to all selected
        # Shape: (num_avail, num_sel)
        dists = compute_dists(avail_pts, sel_pts)
        
        # Check 1: Unique distances from p to selected
        # Sort each row, check for duplicates
        sorted_dists = np.sort(dists, axis=1)
        # Diff along axis 1
        diffs = np.diff(sorted_dists, axis=1)
        # If any diff is 0, duplicate exists
        has_dup = np.any(diffs == 0, axis=1)
        
        # Check 2: Distance matches existing pair distance
        # We need the set of all pairwise squared distances in selected.
        # Let's compute this set.
        # num_sel is small (< 100 usually).
        # K^2 / 2 pairs.
        
        if len(selected_indices) > 1:
            # Compute all pairwise distances in selected
            # sel_pts: (K, 2)
            # dist_sel: (K, K)
            dist_sel = compute_dists(sel_pts, sel_pts)
            # Get upper triangle
            k = len(selected_indices)
            iu = np.triu_indices(k, 1)
            sel_pair_dists = dist_sel[iu]
            sel_pair_dist_set = set(sel_pair_dists.tolist())
            
            # Check if any dist in 'dists' is in sel_pair_dist_set
            # Vectorized check: 
            # Create a boolean mask of same shape as dists
            # np.isin(dists, sel_pair_dist_set)
            # This might be slow if set is large, but K is small.
            has_conflict = np.isin(dists, sel_pair_dist_set)
            has_conflict_any = np.any(has_conflict, axis=1)
        else:
            has_conflict_any = np.zeros(len(available_indices), dtype=bool)
            
        valid_mask = ~has_dup & ~has_conflict_any
        
        if not np.any(valid_mask):
            break
            
        # Get valid indices
        valid_avail_indices = available_indices[valid_mask]
        valid_dists = dists[valid_mask]
        
        # Heuristic: Pick the point that maximizes the minimum distance to selected
        # min_dist_to_sel = np.min(valid_dists, axis=1)
        # best_local_idx = np.argmax(min_dist_to_sel)
        
        # Alternative Heuristic: Pick the point that is furthest from the *centroid* of selected?
        # Or just pick the one with max min-dist.
        
        min_dists = np.min(valid_dists, axis=1)
        best_local_idx = np.argmax(min_dists)
        
        best_idx = valid_avail_indices[best_local_idx]
        
        selected_indices.append(best_idx)
        selected_points = np.vstack([selected_points, points[best_idx]])
        
        # Remove from available
        # Find position of best_idx in available_indices
        # Since available_indices is an array, we can use np.where
        pos = np.where(available_indices == best_idx)[0][0]
        available_indices = np.delete(available_indices, pos)
        
    # Local Search Phase
    # Try to improve the set by swapping points.
    # 1. Identify "bad" points in selected? 
    # 2. Try removing a point p from selected.
    # 3. See if we can add a point q from available (or removed points) that was blocked by p.
    # 4. If len(new_set) > len(old_set), accept.
    
    # To make local search efficient, we need to re-evaluate validity quickly.
    # We will perform a few iterations.
    
    current_set = selected_indices
    # Keep track of all points not in current_set
    # We can just iterate through all points and check if they can be added.
    
    improved = True
    iterations = 0
    max_iterations = 50
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to add any point from the complement
        # This is O(K * N^2) if done naively.
        # We can vectorize:
        # For each point not in current_set, check if it's valid against current_set.
        
        # Create a boolean mask of points in current_set
        in_set = np.zeros(num_points, dtype=bool)
        in_set[current_set] = True
        not_in_set = ~in_set
        candidates = np.where(not_in_set)[0]
        
        if len(candidates) == 0:
            break
            
        cand_pts = points[candidates]
        sel_pts = points[current_set]
        
        # Compute dists: (num_cand, num_sel)
        dists = compute_dists(cand_pts, sel_pts)
        
        # Check validity
        sorted_dists = np.sort(dists, axis=1)
        diffs = np.diff(sorted_dists, axis=1)
        has_dup = np.any(diffs == 0, axis=1)
        
        if len(current_set) > 1:
            dist_sel = compute_dists(sel_pts, sel_pts)
            k = len(current_set)
            iu = np.triu_indices(k, 1)
            sel_pair_dists = dist_sel[iu]
            sel_pair_dist_set = set(sel_pair_dists.tolist())
            has_conflict = np.isin(dists, sel_pair_dist_set)
            has_conflict_any = np.any(has_conflict, axis=1)
        else:
            has_conflict_any = np.zeros(len(candidates), dtype=bool)
            
        valid_mask = ~has_dup & ~has_conflict_any
        
        if np.any(valid_mask):
            # We can add at least one point. 
            # To maximize gain, we might want to add multiple? 
            # But adding one might block others.
            # Let's just add the one that is "best" (e.g., max min-dist) 
            # and restart the loop.
            valid_cands = candidates[valid_mask]
            valid_dists = dists[valid_mask]
            min_dists = np.min(valid_dists, axis=1)
            best_local_idx = np.argmax(min_dists)
            best_idx = valid_cands[best_local_idx]
            
            current_set.append(best_idx)
            improved = True
    
    # Convert indices to (x, y) tuples
    result = [tuple(points[i]) for i in current_set]
    
    return result

# EVOLVE-BLOCK-END
