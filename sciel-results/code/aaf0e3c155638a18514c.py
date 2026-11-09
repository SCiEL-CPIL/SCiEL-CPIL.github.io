import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 330s limit
    
    # Precompute all possible points as a numpy array
    # Shape: (4096, 2)
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    num_points = coords.shape[0]
    
    # We will maintain the current set of points as a list of indices
    # and a numpy array of their coordinates for fast vectorized distance calc
    
    def _try_greedy_fill(current_indices, candidates_order, max_steps=None):
        """
        Attempts to add points from candidates_order to current_indices.
        Returns the updated list of indices.
        """
        current_set = set(current_indices)
        current_coords = coords[current_indices] if current_indices else np.empty((0, 2), dtype=np.int32)
        
        # We need to efficiently check if a candidate can be added.
        # A candidate p can be added if:
        # 1. It is not already in the set.
        # 2. All distances from p to existing points are distinct.
        # 3. For every existing point q, dist(p, q) is not equal to any dist(q, r) for r in set, r!=q.
        
        # Condition 3 is the tricky one. 
        # Let D_q be the set of distances from q to other points in the set.
        # We need to maintain D_q for all q in the set.
        # This is expensive to update if we do it naively.
        
        # Alternative: 
        # Just check the full condition for each candidate?
        # If K is small (~10-20), checking a candidate against K points is O(K^2) if we recompute D_q, 
        # or O(K) if we maintain D_q.
        
        # Let's try to maintain D_q as a list of sets, but updating it for every add/remove is O(K).
        # With K~20, O(K) is fine.
        
        # However, vectorizing the "Check" part is hard if we need to check against D_q.
        # Let's stick to a hybrid: 
        # 1. Use a list of indices.
        # 2. For each candidate, compute distances to all existing points using NumPy.
        # 3. Check uniqueness of these distances (Condition 1).
        # 4. For Condition 2, we need to know if any of these distances conflicts with existing distances at the other end.
        
        # Actually, a simpler sufficient condition for "No Isosceles" is:
        # No three points form an isosceles triangle.
        # This is equivalent to:
        # For every point p in the set, the set of distances to other points has no duplicates.
        
        # Let's optimize the check for a single candidate p against a set S.
        # 1. Compute d_p = dist2(p, s) for all s in S.
        # 2. If len(set(d_p)) < len(d_p): Fail (Isosceles with p as vertex).
        # 3. For each s in S, check if d_p[s] is in the set of distances from s to other points in S.
        #    This requires knowing D_s.
        
        # To make this fast, we will maintain a dictionary `dist_to_others` mapping index -> set of distances.
        # But we are inside a function that might be called many times.
        # Let's restructure: 
        # We will have a global state within the `construct` scope.
        
        pass # Placeholder, will be replaced by the main logic below

    # State
    best_indices = []
    best_size = 0
    
    # We will use a "Kick" strategy:
    # 1. Greedy fill.
    # 2. If stuck, remove k random points.
    # 3. Greedy fill again.
    # 4. Repeat.
    
    # To speed up greedy fill, we will precompute distances between all pairs?
    # 4096x4096 is 16M floats. 128MB. This is acceptable.
    # dist_matrix[i, j] = squared distance between point i and point j.
    # This allows O(1) lookup of distance.
    # Then checking candidate p against set S:
    # d_p = dist_matrix[p, S]
    # Check 1: len(set(d_p)) == len(d_p)
    # Check 2: For each s in S, is d_p[s] in dist_matrix[s, S] \ {d_p[s]}?
    #          Wait, dist_matrix[s, S] includes d_p[s] (distance s-p).
    #          So we need to check if there exists r in S, r!=p, r!=s, such that dist(s, r) == dist(s, p).
    #          This is equivalent to: count of value d_p[s] in dist_matrix[s, S] is > 1?
    #          No, dist_matrix[s, S] contains distances from s to all points in S.
    #          One of them is distance to p, which is d_p[s].
    #          If there is another point r in S such that dist(s, r) == d_p[s], then we have an isosceles triangle (s, p, r).
    #          So we need to check if the value d_p[s] appears more than once in the vector dist_matrix[s, S].
    
    # Let's precompute the distance matrix.
    # Memory: 4096 * 4096 * 4 bytes (int32) = 67MB.
    # Time to compute: 4096^2 / 2 * ~10 ops = ~100M ops. NumPy can do this in <1s.
    
    # Coordinates are (x, y). 
    # dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    # dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    # dist2 = dx*dx + dy*dy
    
    # This creates a 4096x4096 array.
    # Let's do it.
    
    if time.time() > deadline:
        return [(0,0)]
        
    # Compute distance matrix
    # Note: This takes some memory and time.
    # 4096x4096 int32 is ~64MB.
    # Let's try to do it efficiently.
    
    # We can compute row by row to save memory if needed, but 64MB is fine.
    # However, creating the full matrix might take a second or two.
    
    # Let's use a more memory-efficient approach if possible?
    # No, 64MB is fine for modern machines.
    
    # But wait, the evaluator might have limited memory? 
    # Standard limit is usually 2GB+. 64MB is safe.
    
    # Let's compute it.
    # To save time, we can just compute it once.
    
    # Actually, we can avoid the full matrix if we just compute distances on the fly for the current set.
    # The current set size K is small (< 50).
    # Checking a candidate p against K points:
    # 1. Compute d_p (vector of K distances). O(K).
    # 2. Check uniqueness. O(K).
    # 3. For each s in S (K times), check if d_p[s] is duplicated in dist(s, S).
    #    dist(s, S) is a vector of K distances.
    #    Checking if a value is duplicated in a vector of size K is O(K).
    #    Total O(K^2).
    # With K=20, K^2=400.
    # 4096 candidates * 400 ops = 1.6M ops per greedy pass.
    # This is very fast in Python if optimized, or very fast in NumPy.
    
    # Let's stick to the on-the-fly calculation to avoid the 1s+ startup cost of the full matrix,
    # unless the full matrix allows for faster vectorized checks.
    
    # Let's try the on-the-fly NumPy approach.
    
    def _check_add(p_idx, current_indices):
        """
        Returns True if p_idx can be added to current_indices.
        """
        if p_idx in current_indices_set:
            return False
            
        if len(current_indices) == 0:
            return True
            
        p_coord = coords[p_idx]
        # Compute distances from p to all current points
        # current_coords is a numpy array (K, 2)
        # diff = current_coords - p_coord
        # d2 = np.sum(diff*diff, axis=1)
        
        diff = current_coords_arr - p_coord
        d2 = np.sum(diff * diff, axis=1)
        
        # Check 1: Distances from p must be unique
        if len(np.unique(d2)) != len(d2):
            return False
            
        # Check 2: For each point s in current_indices, 
        # the distance d2[s] must not appear elsewhere in the distances from s to other points.
        # We need the set of distances from s to all points in current_indices.
        # This is effectively checking if the triangle (p, s, r) is isosceles with vertex s.
        # i.e., dist(s, p) == dist(s, r) for some r != p.
        
        # We can compute all pairwise distances among current_indices?
        # That's O(K^2).
        # Or we can just check for each s:
        #   dist_s = np.sum((current_coords_arr - coords[s_idx])**2, axis=1)
        #   count of d2[s] in dist_s should be 1 (only the distance to p itself? No, dist_s includes dist to all).
        #   Wait, dist_s is distances from s to ALL points in current_indices.
        #   One of those points is p? No, p is NOT in current_indices yet.
        #   So dist_s contains distances from s to existing points.
        #   We want to know if dist(s, p) is equal to dist(s, r) for any r in current_indices.
        #   dist(s, p) is d2[s].
        #   So we check if d2[s] is in the set of distances from s to current_indices.
        #   But careful: if s is one of the current_indices, the distance from s to itself is 0.
        #   d2[s] is distance from p to s. It is > 0.
        #   So we just check if d2[s] exists in the vector of distances from s to other points in current_indices.
        
        # To do this efficiently, we can precompute a "distance profile" for each point in current_indices?
        # Or just compute it on the fly.
        # K is small.
        
        for i, s_idx in enumerate(current_indices):
            s_coord = coords[s_idx]
            # Distances from s to all current_indices
            diff_s = current_coords_arr - s_coord
            d2_s = np.sum(diff_s * diff_s, axis=1)
            
            # We want to know if d2[s] (dist p-s) is in d2_s.
            # Note: d2_s contains dist(s, s) = 0.
            # d2[s] is dist(p, s).
            # If d2[s] is in d2_s, it means there is some r in current_indices such that dist(s, r) == dist(p, s).
            # If r == s, dist(s, s)=0, but d2[s]>0, so no match.
            # If r != s, then we have an isosceles triangle (p, s, r) with vertex s.
            
            # Efficient check: 
            # Since K is small, we can just use `d2[s] in d2_s`? 
            # `in` on a numpy array is O(K).
            # So total O(K^2).
            
            # To speed this up, we can convert d2_s to a set? 
            # Creating a set of size K is O(K).
            # So O(K) per s, total O(K^2).
            
            # Let's just do the check.
            if d2[s_idx in current_indices] in d2_s: # Wait, index i is the index in the list, not the point index.
                # d2[i] is the distance from p to current_indices[i].
                # d2_s is distances from current_indices[i] to all current_indices.
                if d2[i] in d2_s:
                    return False
                    
        return True

    # The above `_check_add` is O(K^2) with NumPy overhead.
    # Let's optimize by maintaining the distance sets.
    
    # Revised Plan:
    # Maintain:
    # - `current_indices`: list of int
    # - `current_set`: set of int
    # - `dist_sets`: dict {idx: set of distances to other points in current_indices}
    
    # When adding p:
    # 1. Compute d_p = distances from p to all in current_indices. (Vectorized O(K))
    # 2. Check uniqueness of d_p.
    # 3. For each s in current_indices:
    #    Check if d_p[s] is in dist_sets[s].
    
    # This is O(K) per candidate, plus O(K) for the vectorized distance calc.
    # Total O(K) per candidate.
    # 4096 * K. With K=20, ~80k ops. Very fast.
    
    # Let's implement this.
    
    current_indices = []
    current_set = set()
    dist_sets = {}
    current_coords_arr = np.empty((0, 2), dtype=np.int32)
    
    def _add_point(p_idx):
        nonlocal current_indices, current_set, dist_sets, current_coords_arr
        
        if p_idx in current_set:
            return False
            
        if len(current_indices) == 0:
            current_indices.append(p_idx)
            current_set.add(p_idx)
            dist_sets[p_idx] = set()
            current_coords_arr = coords[[p_idx]]
            return True
            
        # Compute distances from p to all current points
        p_coord = coords[p_idx]
        diff = current_coords_arr - p_coord
        d2 = np.sum(diff * diff, axis=1)
        
        # Check 1: Uniqueness
        if len(np.unique(d2)) != len(d2):
            return False
            
        # Check 2: Conflict with existing distance sets
        for i, s_idx in enumerate(current_indices):
            if d2[i] in dist_sets[s_idx]:
                return False
                
        # Valid, add it
        current_indices.append(p_idx)
        current_set.add(p_idx)
        dist_sets[p_idx] = set(d2.tolist())
        
        # Update dist_sets for existing points
        for i, s_idx in enumerate(current_indices[:-1]):
            dist_sets[s_idx].add(d2[i])
            
        # Update current_coords_arr
        # Append row
        current_coords_arr = np.vstack([current_coords_arr, coords[[p_idx]]])
        
        return True

    def _remove_point(p_idx):
        nonlocal current_indices, current_set, dist_sets, current_coords_arr
        
        if p_idx not in current_set:
            return
            
        # Find index in list
        try:
            pos = current_indices.index(p_idx)
        except ValueError:
            return
            
        # Get distances from p to others
        p_coord = coords[p_idx]
        # We need to remove p from current_coords_arr
        # And update dist_sets for all remaining points
        
        # First, compute distances from p to all others (excluding p)
        # current_coords_arr includes p at position pos.
        # Let's just recompute or use the stored dist_sets[p_idx]?
        # dist_sets[p_idx] contains distances to all other points.
        # So we can iterate over current_indices and remove the corresponding distance.
        
        # Remove from current_indices
        current_indices.pop(pos)
        current_set.remove(p_idx)
        
        # Update dist_sets for remaining points
        # For each q in current_indices:
        #   d = dist(p, q)
        #   dist_sets[q].remove(d)
        
        # We need dist(p, q) for all q.
        # We can compute it using the old current_coords_arr?
        # Or just use the fact that we have dist_sets[p_idx] which is a set of distances, but we lost the mapping to which point.
        # So we must compute distances.
        
        # Let's compute distances from p to all points in the NEW current_indices.
        if current_indices:
             p_coord = coords[p_idx]
             # Get coords of remaining points
             remaining_coords = coords[current_indices]
             diff = remaining_coords - p_coord
             d2 = np.sum(diff * diff, axis=1)
             
             for i, q_idx in enumerate(current_indices):
                 dist_sets[q_idx].remove(d2[i])
                 
        # Remove dist_sets[p_idx]
        del dist_sets[p_idx]
        
        # Update current_coords_arr
        # Remove row at pos
        current_coords_arr = np.delete(current_coords_arr, pos, axis=0)
        
    # Main Loop
    rng = random.Random(42)
    
    # Initial Greedy
    candidates = list(range(num_points))
    rng.shuffle(candidates)
    
    for p_idx in candidates:
        if _add_point(p_idx):
            if len(current_indices) > best_size:
                best_size = len(current_indices)
                best_indices = current_indices[:]
    
    # Local Search / Kick
    # While time remains:
    # 1. Try to add random points.
    # 2. If stuck, remove 1-3 random points.
    # 3. Try to add random points again.
    
    stuck_count = 0
    max_stuck = 50
    
    while time.time() < deadline:
        # Try to add
        added_any = False
        
        # Sample random candidates
        # To avoid bias, we can shuffle a small batch or just pick random indices.
        for _ in range(100):
            if time.time() > deadline:
                break
            p_idx = rng.randint(0, num_points - 1)
            if p_idx in current_set:
                continue
            if _add_point(p_idx):
                added_any = True
                if len(current_indices) > best_size:
                    best_size = len(current_indices)
                    best_indices = current_indices[:]
                    
        if added_any:
            stuck_count = 0
        else:
            stuck_count += 1
            
            if stuck_count > max_stuck:
                # Kick: Remove some points
                # Remove 1 to 3 points
                num_remove = rng.randint(1, 3)
                if len(current_indices) > num_remove:
                    to_remove = rng.sample(current_indices, num_remove)
                    for p_idx in to_remove:
                        _remove_point(p_idx)
                    
                    # After removal, try to add again immediately?
                    # The loop will restart and try to add.
                    stuck_count = 0

    return [(i // N, i % N) for i in best_indices]

# EVOLVE-BLOCK-END
