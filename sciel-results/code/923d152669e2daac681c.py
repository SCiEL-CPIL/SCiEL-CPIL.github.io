import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    # We need a set of points in [0, N)^2 such that no three points form an isosceles triangle.
    # An isosceles triangle has at least two equal sides.
    # So for any distinct p, q, r in our set:
    #   dist(p, q) != dist(p, r)
    #   dist(p, q) != dist(q, r)
    #   dist(p, r) != dist(q, r)
    # Equivalently, for any three distinct points, all three pairwise distances must be distinct.

    # Strategy: Simulated Annealing / Local Search
    # State: a subset of grid points.
    # Objective: maximize number of points, subject to validity.
    # Since validity is a hard constraint, we can use a penalty function or repair.
    # Better: Start with a valid set, and try to add points. If adding fails, try swapping.
    
    # Actually, a common approach for "no isosceles" or similar packing problems is:
    # 1. Start with a greedy construction.
    # 2. Use local search: try to add a random point. If it creates an isosceles triangle, 
    #    try to remove one of the conflicting points to free up space, or swap.
    
    # Let's use a more robust method:
    # Maintain a list of points.
    # To check if a new point p can be added:
    #   For every pair (a, b) in current set:
    #     d_pa != d_pb
    #   And for every a in current set:
    #     d_pa != d_ab  (Wait, this is not quite right. The condition is for any triple.)
    
    # Correct condition for adding p to set S:
    # For any distinct a, b in S:
    #   d(p, a) != d(p, b)
    #   d(p, a) != d(a, b)
    #   d(p, b) != d(a, b)
    
    # This is O(k^2) per insertion check where k is current size.
    # With k ~ 10-20 and N=64 (4096 points), this might be slow if done naively for many steps.
    
    # Let's optimize the check.
    # Precompute distances between all grid points? 4096x4096 = 16M entries. 
    # Each entry is an integer (squared distance). 16M * 8 bytes = 128 MB. Acceptable.
    # But we only need distances between selected points and candidates.
    
    # Let's try a simpler heuristic first:
    # 1. Greedy: Pick points one by one that minimize conflicts (or just first valid).
    # 2. Local Search: 
    #    - Try to add a random unselected point. If valid, add it.
    #    - If invalid, identify conflicts. Try to remove a point that is in many conflicts.
    #    - Or, try a swap: remove one selected point, add the new one.
    
    # Let's implement a randomized greedy + local search.
    
    # Precomputation:
    # Squared distances between all pairs of grid points.
    # indices: 0..N*N-1
    # dist[i, j] = squared distance between point i and point j.
    
    points = [(x, y) for y in range(N) for x in range(N)]
    n_points = len(points)
    
    # dist matrix is large. Let's compute on the fly or use a sparse representation?
    # 4096^2 is 16.7 million. np.array of int32 is 67 MB. int16 might overflow (max dist ~ 2*63^2 = 7938). 
    # Max squared dist = (63-0)^2 + (63-0)^2 = 7938. Fits in int16? No, int16 max is 32767. So int16 is safe.
    # But let's use int32 to be safe and fast.
    
    # Actually, computing the full matrix takes time and memory.
    # Alternative: Store points as array of (x,y).
    # When checking if p can be added to S:
    #   For each a in S:
    #     d_pa = dist(p, a)
    #   For each pair a, b in S:
    #     d_ab = dist(a, b)
    #     Check d_pa != d_pb, d_pa != d_ab, d_pb != d_ab.
    
    # This is O(k^2). If k=15, 225 checks. Each check is a few ops.
    # If we do 10,000 steps, that's 2.25 million checks. Very fast in Python if optimized?
    # Python loops are slow. 2M iterations might take ~1-2 seconds.
    # We have 280s. So we can do millions of steps.
    
    # Let's try to be smarter.
    # Maintain for each point p in S, the set of distances to other points in S.
    # When adding p_new:
    #   1. Compute d_new[a] for all a in S.
    #   2. Check if any d_new[a] == d_new[b] for a != b. (Isosceles with p_new as apex)
    #   3. Check if any d_new[a] == d_ab for any a, b in S. (Isosceles with base in S)
    
    # Let's implement this.
    
    # State: list of indices of selected points.
    # selected = set()
    
    # Helper: can_add(p_idx)
    #   if not selected: return True
    #   d_to_sel = {}
    #   for a in selected:
    #       d = dist(p_idx, a)
    #       if d in d_to_sel: return False # Two points in S at same distance from p
    #       d_to_sel[d] = a
    #   # Now check against existing distances in S
    #   # We need to know all pairwise distances in S.
    #   # This is expensive to maintain.
    
    # Alternative: Just check all triples involving p_new.
    # For each a in S:
    #   d_pa = dist(p, a)
    #   For each b in S, b != a:
    #       d_ab = dist(a, b)
    #       d_pb = dist(p, b)
    #       if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab: return False
    # This is O(k^2).
    
    # Let's try a different strategy:
    # Start with a known good configuration?
    # The seed program found ~10 points on a line.
    # Can we do better in 2D?
    
    # Let's use Simulated Annealing.
    # State: set of points.
    # Move: Add a random point, Remove a random point, or Swap.
    # Cost: Number of isosceles triangles.
    # We want to minimize cost to 0, and maximize size.
    # Objective: -size * 1000 + cost
    # Or: If cost > 0, penalize heavily.
    
    # Let's define:
    # energy = cost * 10000 - len(selected)
    # We want to minimize energy.
    
    # To compute cost efficiently:
    # When adding p:
    #   new_cost = 0
    #   for a in S:
    #     d_pa = dist(p, a)
    #     for b in S, b != a:
    #        d_ab = dist(a, b)
    #        d_pb = dist(p, b)
    #        if d_pa == d_pb: new_cost += 1
    #        if d_pa == d_ab: new_cost += 1
    #        if d_pb == d_ab: new_cost += 1
    #   Note: Each isosceles triangle is counted once?
    #   Triangle (p, a, b) is isosceles if any two sides equal.
    #   If d_pa == d_pb, it's isosceles.
    #   If d_pa == d_ab, it's isosceles.
    #   If d_pb == d_ab, it's isosceles.
    #   It's possible for two conditions to be true (equilateral? No, grid points rarely form equilateral triangles with integer coords unless specific. Even then, it would be counted multiple times).
    #   We just need cost > 0 to be invalid.
    
    # This O(k^2) check is the bottleneck.
    # Let's optimize by precomputing distances between all pairs of grid points?
    # 4096x4096 matrix of int16.
    # Memory: 4096*4096*2 = 32 MB.
    # Time to compute: 4096*4096/2 = 8M pairs.
    # This is feasible.
    
    # Let's do that.
    
    # 1. Create index to (x,y) mapping.
    # 2. Compute dist_matrix[i, j] = (xi-xj)^2 + (yi-yj)^2.
    
    # Then, can_add(p, S) can use dist_matrix[p, a] and dist_matrix[a, b].
    
    # Let's implement this.
    
    # Step 1: Precompute distances
    # This might take a few seconds.
    
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    coords = np.column_stack((X.ravel(), Y.ravel()))
    n_pts = len(coords)
    
    # Compute squared distances
    # dist[i, j] = |coords[i] - coords[j]|^2
    # Using broadcasting is memory heavy for 4096x4096x2.
    # Let's compute in chunks or use a loop?
    # Or:
    # dist = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1)
    # This creates a 4096x4096x2 array. 4096*4096*2*8 = 256 MB. Might be okay.
    
    # Let's try it.
    if time.time() > deadline - 60:
        return [(0,0)] # Fallback
    
    # Compute distance matrix
    # To save memory, we can compute row by row if needed, but let's try full matrix first.
    # If it's too slow or memory heavy, we'll switch to on-the-fly.
    
    # Actually, 4096 is small enough.
    dist_mat = np.zeros((n_pts, n_pts), dtype=np.int32)
    # Vectorized calculation
    # coords shape (n_pts, 2)
    # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :] # (n_pts, n_pts, 2)
    # dist_mat = np.sum(diff**2, axis=2)
    
    # This might be slow in pure Python/NumPy due to memory allocation.
    # Let's do it in a loop to be safe on memory, or just accept the time.
    # 4096^2 = 16M operations. NumPy is fast.
    
    # Let's try:
    t0 = time.time()
    # Use a more memory efficient method?
    # Actually, we can compute distances on the fly using the coordinates.
    # coords[a] - coords[b]
    
    # Let's stick to on-the-fly distance calculation to avoid the 256MB temporary array.
    # It's just 2 subtractions and 2 multiplications per pair.
    
    def get_dist(i, j):
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        return dx*dx + dy*dy
    
    # This is a Python function call, which is slow.
    # Let's precompute the distance matrix using NumPy, but carefully.
    
    # Let's try the vectorized way. If it fails or is too slow, we'll know.
    try:
        diff = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.sum(diff**2, axis=-1)
        del diff
    except MemoryError:
        # Fallback to loop
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = get_dist(i, j)
                dist_mat[i, j] = d
                dist_mat[j, i] = d
    
    # Now we have dist_mat.
    
    # Simulated Annealing
    # State: selected (set of indices)
    # We want to maximize size, subject to no isosceles triangles.
    
    # Initialize with greedy
    selected = set()
    order = list(range(n_pts))
    random.shuffle(order)
    
    for p in order:
        if time.time() > deadline - 30:
            break
        # Check if p can be added
        valid = True
        d_to_sel = {}
        for a in selected:
            d = dist_mat[p, a]
            if d in d_to_sel:
                valid = False
                break
            d_to_sel[d] = a
        
        if valid:
            # Check against existing pairs
            # This part is O(k^2)
            # We need to check if d_pa == d_ab for any a, b in selected.
            # We can precompute a set of all pairwise distances in selected?
            # No, because d_ab depends on the pair.
            # We just have to check.
            for a in selected:
                d_pa = dist_mat[p, a]
                for b in selected:
                    if b == a: continue
                    d_ab = dist_mat[a, b]
                    d_pb = dist_mat[p, b]
                    if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            selected.add(p)
    
    best_size = len(selected)
    best_selected = set(selected)
    
    # Local Search / Simulated Annealing
    # We will try to improve the set.
    # Move: 
    #   1. Add a random point.
    #   2. Remove a random point.
    #   3. Swap: Remove one, add another.
    
    # To make it fast, we only consider moves that might improve or are neutral.
    
    current_selected = set(selected)
    current_size = len(current_selected)
    
    # Precompute a list of all points not in selected?
    # We can just pick random indices.
    
    for step in range(100000): # Max steps
        if time.time() > deadline:
            break
        
        # Random move
        move_type = random.random()
        if move_type < 0.5:
            # Add
            p = random.randint(0, n_pts - 1)
            if p in current_selected:
                continue
            
            # Check validity of adding p
            # This is the expensive part.
            # Let's see if we can speed it up.
            # We need to check p against all pairs in current_selected.
            
            valid = True
            d_to_sel = {}
            for a in current_selected:
                d = dist_mat[p, a]
                if d in d_to_sel:
                    valid = False
                    break
                d_to_sel[d] = a
            
            if valid:
                for a in current_selected:
                    d_pa = dist_mat[p, a]
                    for b in current_selected:
                        if b == a: continue
                        d_ab = dist_mat[a, b]
                        d_pb = dist_mat[p, b]
                        if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                current_selected.add(p)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_selected = set(current_selected)
        
        elif move_type < 0.75:
            # Remove
            if not current_selected:
                continue
            p = random.choice(list(current_selected))
            current_selected.remove(p)
            current_size -= 1
        
        else:
            # Swap
            if not current_selected:
                continue
            p_remove = random.choice(list(current_selected))
            p_add = random.randint(0, n_pts - 1)
            if p_add in current_selected:
                continue
            
            # Try the swap
            # Temporarily remove p_remove
            current_selected.remove(p_remove)
            
            # Check if p_add can be added
            valid = True
            d_to_sel = {}
            for a in current_selected:
                d = dist_mat[p_add, a]
                if d in d_to_sel:
                    valid = False
                    break
                d_to_sel[d] = a
            
            if valid:
                for a in current_selected:
                    d_pa = dist_mat[p_add, a]
                    for b in current_selected:
                        if b == a: continue
                        d_ab = dist_mat[a, b]
                        d_pb = dist_mat[p_add, b]
                        if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                current_selected.add(p_add)
                # current_size stays same
            else:
                # Revert
                current_selected.add(p_remove)
        
        # Cooling? We don't really have a temperature in this simple version.
        # We accept improving moves (add) and neutral moves (swap, remove) randomly?
        # This is more like Random Walk with Restart.
        
        # To escape local optima, we should sometimes accept worse moves (remove).
        # The above logic always accepts add if valid, always accepts remove, and accepts swap if valid.
        # This might get stuck.
        
        # Let's add a restart mechanism.
        # If we haven't improved in X steps, restart with a new greedy initialization.
        
        if step % 1000 == 0 and current_size < best_size:
            # Restart
            selected = set()
            order = list(range(n_pts))
            random.shuffle(order)
            for p in order:
                if time.time() > deadline - 10:
                    break
                valid = True
                d_to_sel = {}
                for a in selected:
                    d = dist_mat[p, a]
                    if d in d_to_sel:
                        valid = False
                        break
                    d_to_sel[d] = a
                if valid:
                    for a in selected:
                        d_pa = dist_mat[p, a]
                        for b in selected:
                            if b == a: continue
                            d_ab = dist_mat[a, b]
                            d_pb = dist_mat[p, b]
                            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    selected.add(p)
            
            current_selected = set(selected)
            current_size = len(current_selected)
    
    # Convert best_selected to list of (x, y)
    result = [tuple(coords[i]) for i in best_selected]
    return result

# To ensure we don't hang, we can run multiple independent runs if time permits?
# The above loop runs until deadline.

# Let's refine the local search to be more effective.
# The current implementation is a bit slow due to Python loops in the validity check.
# We can optimize the validity check by precomputing distances to selected points.

# Let's try a different approach:
# Use a bitset or array to track selected points.
# And precompute for each point p, the distances to all other points.
# This is what dist_mat is.

# The bottleneck is the triple loop in validity check.
# Can we vectorize it?
# For a candidate p and set S:
# d_p = dist_mat[p, :] # vector of distances from p to all points
# S_indices = np.array(list(S))
# d_p_S = d_p[S_indices]
# 
# Check 1: Are there duplicates in d_p_S?
# if len(set(d_p_S)) < len(d_p_S): invalid
# 
# Check 2: For each pair a, b in S:
#   d_ab = dist_mat[a, b]
#   if d_ab in d_p_S: invalid?
#   Wait, the condition is:
#   d(p, a) != d(p, b)
#   d(p, a) != d(a, b)
#   d(p, b) != d(a, b)
#   
#   So for a fixed pair (a, b) in S:
#   d_ab must not be in d_p_S.
#   AND d_p[a] != d_p[b].
#   
#   So:
#   1. Check duplicates in d_p_S.
#   2. Compute all pairwise distances in S: D_S = {dist_mat[a, b] for a, b in S, a < b}
#   3. Check if any value in D_S is in d_p_S.
#   
#   This is much faster!
#   Step 1: O(k)
#   Step 2: O(k^2) to compute D_S, but we can maintain it incrementally?
#   Step 3: O(k^2) to check intersection.
#   
#   If we maintain D_S (set of all pairwise distances in S), updating it when adding/removing is O(k).
#   Then adding p is:
#   1. Compute d_p_S (O(k))
#   2. Check duplicates in d_p_S (O(k))
#   3. Check intersection of D_S and set(d_p_S) (O(k) if D_S is a set)
#   
#   This is O(k) per insertion check!
#   This is a huge improvement.

def construct_optimized():
    start_time = time.time()
    deadline = start_time + 280.0

    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    coords = np.column_stack((X.ravel(), Y.ravel()))
    n_pts = len(coords)

    # Precompute distance matrix
    try:
        diff = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.sum(diff**2, axis=-1)
        del diff
    except MemoryError:
        dist_mat = np.zeros((n_pts, n_pts), dtype=np.int32)
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                d = dx*dx + dy*dy
                dist_mat[i, j] = d
                dist_mat[j, i] = d

    # State:
    # selected: set of indices
    # dist_set: set of all pairwise squared distances among selected points
    
    def can_add(p, selected, dist_set):
        if not selected:
            return True
        
        # Get distances from p to all in selected
        sel_list = list(selected)
        d_p_S = dist_mat[p, sel_list]
        
        # Check duplicates
        if len(set(d_p_S)) < len(d_p_S):
            return False
        
        # Check if any d in d_p_S is in dist_set
        d_set = set(d_p_S)
        if d_set & dist_set:
            return False
        
        return True

    def add_point(p, selected, dist_set):
        # Add p to selected and update dist_set
        # New distances: dist_mat[p, a] for a in selected
        for a in selected:
            dist_set.add(dist_mat[p, a])
        selected.add(p)

    def remove_point(p, selected, dist_set):
        # Remove p from selected and update dist_set
        # This is tricky because we need to know which distances to remove.
        # We can recompute dist_set from scratch? O(k^2).
        # Or maintain a count of each distance?
        # Let's just recompute for simplicity if k is small.
        selected.remove(p)
        # Recompute dist_set
        dist_set.clear()
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i+1, len(sel_list)):
                dist_set.add(dist_mat[sel_list[i], sel_list[j]])

    # Greedy Initialization
    selected = set()
    dist_set = set()
    order = list(range(n_pts))
    random.shuffle(order)
    
    for p in order:
        if time.time() > deadline - 30:
            break
        if can_add(p, selected, dist_set):
            add_point(p, selected, dist_set)
    
    best_size = len(selected)
    best_selected = set(selected)
    best_dist_set = set(dist_set)
    
    # Local Search
    current_selected = set(selected)
    current_dist_set = set(dist_set)
    current_size = len(current_selected)
    
    # We will use a simple hill climbing with random restarts.
    # Move: Add, Remove, Swap.
    
    for step in range(50000):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Add
            p = random.randint(0, n_pts - 1)
            if p in current_selected:
                continue
            if can_add(p, current_selected, current_dist_set):
                add_point(p, current_selected, current_dist_set)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_selected = set(current_selected)
                    best_dist_set = set(current_dist_set)
        
        elif move_type < 0.75:
            # Remove
            if not current_selected:
                continue
            p = random.choice(list(current_selected))
            remove_point(p, current_selected, current_dist_set)
            current_size -= 1
        
        else:
            # Swap
            if not current_selected:
                continue
            p_remove = random.choice(list(current_selected))
            p_add = random.randint(0, n_pts - 1)
            if p_add in current_selected:
                continue
            
            # Try swap
            # Save state
            old_selected = set(current_selected)
            old_dist_set = set(current_dist_set)
            
            remove_point(p_remove, current_selected, current_dist_set)
            
            if can_add(p_add, current_selected, current_dist_set):
                add_point(p_add, current_selected, current_dist_set)
                # current_size unchanged
            else:
                # Revert
                current_selected = old_selected
                current_dist_set = old_dist_set
        
        # Restart if stuck
        if step % 500 == 0 and current_size < best_size:
            # Restart with new greedy
            current_selected = set()
            current_dist_set = set()
            order = list(range(n_pts))
            random.shuffle(order)
            for p in order:
                if time.time() > deadline - 10:
                    break
                if can_add(p, current_selected, current_dist_set):
                    add_point(p, current_selected, current_dist_set)
            current_size = len(current_selected)
    
    result = [tuple(coords[i]) for i in best_selected]
    return result

# Use the optimized version
def construct():
    return construct_optimized()
