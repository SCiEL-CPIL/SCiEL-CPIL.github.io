import numpy as np
import random
import time
from itertools import combinations

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    # All candidate points in the grid
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_total = len(all_points)
    
    # Precompute squared distances between all pairs
    # dist[i, j] = squared distance between point i and point j
    pts = np.array(all_points, dtype=np.float64)
    # Compute pairwise squared distances
    # diff = pts[:, None, :] - pts[None, :, :]
    # dist = np.sum(diff**2, axis=2)
    # This is 4096x4096 = 16M entries, should be fine memory-wise (128MB)
    
    # Actually, let's be smarter. We'll compute on-the-fly but cache.
    # For a set of selected points, we need to check:
    # For any three points a, b, c in the set:
    #   dist(a,c) != dist(b,c)  (c is not equidistant from a and b)
    #   dist(a,b) != dist(a,c)  (a is not equidistant from b and c)
    #   dist(a,b) != dist(b,c)  (b is not equidistant from a and c)
    # In other words, no isosceles triangle.
    
    # A point c conflicts with pair (a,b) if:
    #   dist(c,a) == dist(c,b)  OR
    #   dist(c,a) == dist(a,b)  OR
    #   dist(c,b) == dist(a,b)
    
    # Strategy: Greedy with random restarts + local improvement
    # Use incremental validity checking with sets of distances
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy
    
    def can_add_point(p_new, selected, dist_cache):
        """Check if p_new can be added to selected without creating isosceles triangle.
        dist_cache maps (i,j) -> squared distance for i<j in selected indices
        """
        n_sel = len(selected)
        for i in range(n_sel):
            a = selected[i]
            da = sq_dist(p_new, a)
            for j in range(i+1, n_sel):
                b = selected[j]
                db = sq_dist(p_new, b)
                ab = sq_dist(a, b)
                # Check if triangle (a, b, p_new) is isosceles
                if da == db or da == ab or db == ab:
                    return False
        return True
    
    def greedy_random(order, deadline):
        """Greedy construction following a random order."""
        selected = []
        # For efficiency, maintain set of points that are "blocked" by each selected point
        # Actually, let's just do the O(n^2) check per candidate
        
        for p in order:
            if time.time() > deadline - 5.0:
                break
            if can_add_point(p, selected, None):
                selected.append(p)
        return selected
    
    def local_improve(selected, deadline):
        """Try to improve the set by swapping or adding points."""
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try adding each unselected point
            for p in all_points:
                if p in selected_set:
                    continue
                if can_add_point(p, selected, None):
                    selected.append(p)
                    selected_set.add(p)
                    improved = True
                    break  # Restart scan after improvement
            # Try swapping: remove one point, try to add two
            if not improved:
                for i in range(len(selected)):
                    if time.time() > deadline - 5.0:
                        break
                    saved = selected.pop(i)
                    selected_set.discard(saved)
                    added = 0
                    for p in all_points:
                        if p in selected_set:
                            continue
                        if can_add_point(p, selected, None):
                            selected.append(p)
                            selected_set.add(p)
                            added += 1
                            if added >= 2:
                                break
                    if added < 1:
                        # Revert
                        selected.append(saved)
                        selected_set.add(saved)
                    else:
                        improved = True
                        break
        return selected
    
    best_set = []
    best_len = 0
    
    # Multiple random restarts
    rng = random.Random(42)
    n_restarts = 500
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random order
        order = all_points[:]
        rng.shuffle(order)
        
        # Greedy
        current = greedy_random(order, deadline)
        current_len = len(current)
        
        if current_len > best_len:
            best_set = current[:]
            best_len = current_len
        
        # Local improvement
        if current_len >= best_len - 2:  # Only improve if close to best
            improved = local_improve(current, deadline)
            if len(improved) > best_len:
                best_set = improved
                best_len = len(improved)
    
    return best_set

# Precompute for speed: use numpy for distance checks
# Actually, the above is pure Python which may be slow. Let's optimize.

# Redesign: use numpy-based validity checking

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    N_local = N
    all_points = [(x, y) for x in range(N_local) for y in range(N_local)]
    n_total = len(all_points)
    
    # Convert to numpy for vectorized distance computation
    pts_arr = np.array(all_points, dtype=np.int32)
    
    # Precompute all pairwise squared distances
    # dist_mat[i, j] = squared distance between point i and point j
    diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
    dist_mat = np.sum(diff.astype(np.int64) ** 2, axis=2)
    del diff
    
    # dist_mat is symmetric, dist_mat[i,i] = 0
    
    def can_add_vectorized(p_idx, selected_indices):
        """Check if point p_idx can be added to selected_indices.
        selected_indices: list of indices into all_points
        """
        if len(selected_indices) < 2:
            return True
        
        sel_arr = np.array(selected_indices, dtype=np.int32)
        # dists from p to all selected
        d_p_sel = dist_mat[p_idx, sel_arr]  # shape (k,)
        # dists among selected
        d_sel_sel = dist_mat[np.ix_(sel_arr, sel_arr)]  # shape (k, k)
        
        # For each pair (i, j) in selected, check if triangle (p, sel_i, sel_j) is isosceles
        # d_p_sel[i] == d_p_sel[j]  OR  d_p_sel[i] == d_sel_sel[i,j]  OR  d_p_sel[j] == d_sel_sel[i,j]
        
        k = len(sel_arr)
        # Build upper triangle indices
        for i in range(k):
            for j in range(i+1, k):
                if d_p_sel[i] == d_p_sel[j] or d_p_sel[i] == d_sel_sel[i, j] or d_p_sel[j] == d_sel_sel[i, j]:
                    return False
        return True
    
    def greedy_random(order, deadline):
        selected = []
        for p_idx in order:
            if time.time() > deadline - 5.0:
                break
            if can_add_vectorized(p_idx, selected):
                selected.append(p_idx)
        return selected
    
    def local_improve(selected, deadline):
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try adding each unselected point
            for p_idx in range(n_total):
                if p_idx in selected_set:
                    continue
                if can_add_vectorized(p_idx, selected):
                    selected.append(p_idx)
                    selected_set.add(p_idx)
                    improved = True
                    break
        return selected
    
    best_set = []
    best_len = 0
    
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > deadline - 10.0:
            break
        
        order = list(range(n_total))
        rng.shuffle(order)
        
        current = greedy_random(order, deadline)
        current_len = len(current)
        
        if current_len > best_len:
            best_set = current[:]
            best_len = current_len
        
        if current_len >= best_len - 2:
            improved = local_improve(current, deadline)
            if len(improved) > best_len:
                best_set = improved
                best_len = len(improved)
    
    return [all_points[i] for i in best_set]
