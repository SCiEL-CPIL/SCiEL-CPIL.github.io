"""
No-isosceles set: maximize points in [0,N)^2 with no isosceles triangle.
Strategy: Vectorized constraint checking using NumPy to batch-process isosceles violations.
This allows for a faster "add" phase and a high-throughput local search (simulated annealing) phase.
"""
import numpy as np
import random
import time

N = 64
TIME_LIMIT = 280.0

# Precompute coordinates for vectorization
xs = np.arange(N)
ys = np.arange(N)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
COORDS = np.stack([XX.ravel(), YY.ravel()], axis=1)
TOTAL_POINTS = N * N

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 5.0

    # 1. Greedy Initialization (Vectorized)
    # We try to build a large set quickly by checking candidates in batches.
    # However, simple batch checking is tricky because adding one point invalidates others.
    # We use a heuristic: pick random points, check against current set, keep valid ones.
    # To speed this up, we don't check every point against every other point in a python loop,
    # but we do have to be careful.
    
    # Let's use a faster greedy: 
    # Start with empty set.
    # Randomly permute all points.
    # For each point, check if it violates isosceles with current set.
    # The check for a single point P against set S:
    #   For all p1, p2 in S: dist(P, p1) != dist(P, p2) AND dist(P, p1) != dist(p1, p2) AND dist(P, p2) != dist(p1, p2)
    #   This is O(|S|^2) per point if done naively.
    #   
    # Alternative: 
    #   Maintain a set of "forbidden" points? No, isosceles is a 3-point property.
    #   
    # Let's stick to a Python loop for the greedy phase but optimize the check.
    # Actually, the current program's score is 9.25. We need to beat that.
    # The bottleneck is usually the local search.
    
    # Let's try a different initialization: 
    # Use a grid subsample or a random set and prune it?
    # Pruning is hard.
    
    # Let's use the vectorized check for the LOCAL SEARCH which is the main driver for improvement.
    
    # Initialization: Random Greedy
    best_set = []
    best_size = 0
    
    # Try a few random greedy starts
    for _ in range(5):
        if time.time() > deadline - 100: break
        
        current_set = []
        current_set_coords = np.array([]).reshape(0, 2)
        
        # Random order
        order = np.random.permutation(TOTAL_POINTS)
        
        # This loop is O(N^2 * K) which is slow. 
        # Let's try to add points in a smarter way.
        # How about we just pick a random subset of size K and check if it's valid?
        # If not, remove the most violating points?
        
        # Let's stick to the greedy but optimize the check function.
        def is_valid_point(p, set_coords):
            """
            Check if point p can be added to set_coords without forming an isosceles triangle.
            p: (x, y)
            set_coords: (K, 2) array
            """
            if len(set_coords) < 2:
                return True
            
            # Distances from p to all points in set
            dx = set_coords[:, 0] - p[0]
            dy = set_coords[:, 1] - p[1]
            dists_to_p = dx**2 + dy**2
            
            # Check 1: No two points in set are at the same distance from p
            # If there are duplicates in dists_to_p, it's invalid.
            if len(np.unique(dists_to_p)) != len(dists_to_p):
                return False
            
            # Check 2: For every pair (i, j) in set, dist(p, i) != dist(i, j) and dist(p, j) != dist(i, j)
            # This is O(K^2). We can vectorize the pair distances.
            # dist(i, j) for all i, j.
            # Let's compute the matrix of squared distances between points in the set.
            # But we only need to check against dists_to_p.
            
            # For each i in set:
            #   dist(p, i) is dists_to_p[i]
            #   We need to ensure no j exists such that dist(i, j) == dist(p, i)
            #   AND no j exists such that dist(i, j) == dist(p, j)
            
            # This is still O(K^2).
            # Let's just do the loop in Python for now, it might be fast enough for K~100.
            K = len(set_coords)
            for i in range(K):
                di = dists_to_p[i]
                pi = set_coords[i]
                for j in range(i + 1, K):
                    dj = dists_to_p[j]
                    pj = set_coords[j]
                    dx_ij = pi[0] - pj[0]
                    dy_ij = pi[1] - pj[1]
                    dij = dx_ij*dx_ij + dy_ij*dy_ij
                    if di == dij or dj == dij:
                        return False
            return True

        # This is still slow. Let's use the vectorized approach for the check itself.
        # We can precompute the distance matrix for the current set?
        # No, the set changes.
        
        # Let's try a different strategy for initialization:
        # Start with a sparse grid, then add points.
        
        # Actually, let's just run the local search on a decent random start.
        # Random start of size 50?
        # Check if valid? If not, prune?
        
        # Let's use the existing greedy logic but with a time check.
        current = []
        current_coords = np.array([]).reshape(0, 2)
        
        for idx in order:
            if time.time() > deadline - 100: break
            p = COORDS[idx]
            # Fast check: 
            # We can check the "apex" condition (dist to p) quickly with numpy.
            if len(current_coords) > 0:
                dx = current_coords[:, 0] - p[0]
                dy = current_coords[:, 1] - p[1]
                dists = dx**2 + dy**2
                if len(np.unique(dists)) != len(dists):
                    continue # Apex violation
            
            # Now check base condition. This is the expensive part.
            # We'll do it in a loop for now, but only for points that pass the apex check.
            valid = True
            K = len(current_coords)
            for i in range(K):
                di = dists[i]
                pi = current_coords[i]
                # We need to check if dist(pi, pj) == di for any j != i
                # And dist(pi, pj) == dists[j] for any j != i
                
                # Vectorized check for fixed i:
                # dist(pi, all others)
                dx_i = current_coords[:, 0] - pi[0]
                dy_i = current_coords[:, 1] - pi[1]
                dists_i = dx_i**2 + dy_i**2
                
                # Exclude i itself
                mask = np.ones(K, dtype=bool)
                mask[i] = False
                dists_i_others = dists_i[mask]
                dists_p_others = dists[mask]
                
                # Check if di is in dists_i_others
                if di in dists_i_others:
                    valid = False
                    break
                # Check if any dists_p_others[j] equals dists_i_others[j]
                # Wait, the condition is: 
                # Triangle (p, pi, pj) is isosceles if:
                # 1. dist(p, pi) == dist(p, pj)  (Handled by apex check above, but that was for ALL pairs. 
                #    If apex check passed, dist(p, pi) != dist(p, pj) for all i != j. So this is safe.)
                # 2. dist(p, pi) == dist(pi, pj)  => di == dist(pi, pj)
                # 3. dist(p, pj) == dist(pi, pj)  => dist(p, pj) == dist(pi, pj)
                
                # So for fixed i, we need to check if there exists j such that:
                # dist(pi, pj) == di  OR  dist(pi, pj) == dist(p, pj)
                
                # dists_i_others contains dist(pi, pj) for all j != i.
                # dists_p_others contains dist(p, pj) for all j != i.
                
                # Check 2: di in dists_i_others
                if np.any(dists_i_others == di):
                    valid = False
                    break
                
                # Check 3: dist(pi, pj) == dist(p, pj)
                # This means dists_i_others == dists_p_others element-wise?
                # No, it means for some j, dist(pi, pj) == dist(p, pj).
                # So we check if there is any index k in the masked arrays where they are equal.
                if np.any(dists_i_others == dists_p_others):
                    valid = False
                    break
            
            if valid:
                current.append(p)
                current_coords = np.vstack([current_coords, p])
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current

    # 2. Local Search (Simulated Annealing)
    # We use the vectorized check to evaluate swaps quickly.
    
    if best_size == 0:
        return []
        
    current_set = list(best_set)
    current_set_set = set(current_set)
    
    # Precompute coordinates of current set
    current_coords = np.array(current_set)
    
    # List of points NOT in the set
    all_indices = np.arange(TOTAL_POINTS)
    in_set_indices = np.array([int(c[0]) * N + int(c[1]) for c in current_set])
    not_in_set_mask = np.ones(TOTAL_POINTS, dtype=bool)
    not_in_set_mask[in_set_indices] = False
    not_in_set_indices = all_indices[not_in_set_mask]
    
    temperature = 10.0
    cooling_rate = 0.99
    min_temperature = 0.1
    
    # Helper to check if a point can be added to a given set of coordinates
    # Returns True if valid.
    def check_point_validity(p, set_coords):
        if len(set_coords) < 2:
            return True
        dx = set_coords[:, 0] - p[0]
        dy = set_coords[:, 1] - p[1]
        dists = dx**2 + dy**2
        
        if len(np.unique(dists)) != len(dists):
            return False
        
        K = len(set_coords)
        for i in range(K):
            di = dists[i]
            pi = set_coords[i]
            dx_i = set_coords[:, 0] - pi[0]
            dy_i = set_coords[:, 1] - pi[1]
            dists_i = dx_i**2 + dy_i**2
            
            mask = np.ones(K, dtype=bool)
            mask[i] = False
            dists_i_others = dists_i[mask]
            dists_p_others = dists[mask]
            
            if np.any(dists_i_others == di):
                return False
            if np.any(dists_i_others == dists_p_others):
                return False
        return True

    # Main Loop
    while time.time() < deadline:
        # Pick a random point from the set to remove
        i = random.randint(0, len(current_set) - 1)
        p_remove = current_set[i]
        
        # Pick a random point from the not-in-set to add
        # To be efficient, we might want to pick from "promising" candidates, but random is okay for SA.
        if len(not_in_set_indices) == 0:
            break
        j_idx = random.randint(0, len(not_in_set_indices) - 1)
        p_add_idx = not_in_set_indices[j_idx]
        p_add = COORDS[p_add_idx]
        
        # Perform swap
        # Remove p_remove
        temp_set = [p for k, p in enumerate(current_set) if k != i]
        temp_coords = np.array(temp_set)
        
        # Check if p_add is valid in temp_set
        if check_point_validity(p_add, temp_coords):
            # Accept the swap
            current_set[i] = p_add
            current_set_set.remove(p_remove)
            current_set_set.add(p_add)
            current_coords = np.array(current_set)
            
            # Update not_in_set_indices
            # This is O(N^2) to rebuild. We can do it lazily or with a set.
            # Let's just rebuild if the set size is small, or use a different structure.
            # For now, let's just check membership in the set.
            # Actually, rebuilding not_in_set_indices every step is too slow.
            # We'll just pick a random point from the entire grid and check if it's in the set.
            # If it is, we skip or pick another.
            
            # Better: maintain a list of candidates.
            # But for simplicity in this code block, let's just pick random indices and verify.
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
        
        # Cooling
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature

    return best_set

# To make the local search more effective, we should probably not just do random swaps.
# We should try to ADD points if possible.
# And we should update the "not_in_set" list more efficiently.

# Let's refine the local search to include "add" moves.

def construct_refined():
    start = time.time()
    deadline = start + TIME_LIMIT - 5.0

    # 1. Greedy Initialization
    best_set = []
    best_size = 0
    
    for _ in range(3):
        if time.time() > deadline - 150: break
        
        current = []
        current_coords = np.array([]).reshape(0, 2)
        order = np.random.permutation(TOTAL_POINTS)
        
        for idx in order:
            if time.time() > deadline - 150: break
            p = COORDS[idx]
            
            if len(current_coords) > 0:
                dx = current_coords[:, 0] - p[0]
                dy = current_coords[:, 1] - p[1]
                dists = dx**2 + dy**2
                if len(np.unique(dists)) != len(dists):
                    continue
            
            valid = True
            K = len(current_coords)
            for i in range(K):
                di = dists[i]
                pi = current_coords[i]
                dx_i = current_coords[:, 0] - pi[0]
                dy_i = current_coords[:, 1] - pi[1]
                dists_i = dx_i**2 + dy_i**2
                
                mask = np.ones(K, dtype=bool)
                mask[i] = False
                dists_i_others = dists_i[mask]
                dists_p_others = dists[mask]
                
                if np.any(dists_i_others == di):
                    valid = False
                    break
                if np.any(dists_i_others == dists_p_others):
                    valid = False
                    break
            
            if valid:
                current.append(p)
                current_coords = np.vstack([current_coords, p])
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

    if best_size == 0:
        return []

    # 2. Local Search with Add and Swap
    current_set = list(best_set)
    current_set_set = set(current_set)
    
    # We will use a list of "available" points.
    # To avoid O(N^2) rebuilds, we'll just pick random points from the grid and check if they are in the set.
    # If they are in the set, we consider them for removal (swap).
    # If they are not, we consider them for addition.
    
    temperature = 10.0
    cooling_rate = 0.995
    min_temperature = 0.1
    
    def check_point_validity(p, set_coords):
        if len(set_coords) < 2:
            return True
        dx = set_coords[:, 0] - p[0]
        dy = set_coords[:, 1] - p[1]
        dists = dx**2 + dy**2
        
        if len(np.unique(dists)) != len(dists):
            return False
        
        K = len(set_coords)
        for i in range(K):
            di = dists[i]
            pi = set_coords[i]
            dx_i = set_coords[:, 0] - pi[0]
            dy_i = set_coords[:, 1] - pi[1]
            dists_i = dx_i**2 + dy_i**2
            
            mask = np.ones(K, dtype=bool)
            mask[i] = False
            dists_i_others = dists_i[mask]
            dists_p_others = dists[mask]
            
            if np.any(dists_i_others == di):
                return False
            if np.any(dists_i_others == dists_p_others):
                return False
        return True

    while time.time() < deadline:
        # Randomly pick a point from the grid
        p_rand = COORDS[random.randint(0, TOTAL_POINTS - 1)]
        
        if p_rand in current_set_set:
            # It's in the set. Try to swap it out.
            # Pick a random point NOT in the set.
            p_new = None
            attempts = 0
            while attempts < 10:
                candidate = COORDS[random.randint(0, TOTAL_POINTS - 1)]
                if candidate not in current_set_set:
                    p_new = candidate
                    break
                attempts += 1
            
            if p_new is None:
                continue
                
            # Try swap: remove p_rand, add p_new
            temp_set = [p for p in current_set if p != p_rand]
            temp_coords = np.array(temp_set)
            
            if check_point_validity(p_new, temp_coords):
                # Accept
                current_set.remove(p_rand)
                current_set.append(p_new)
                current_set_set.remove(p_rand)
                current_set_set.add(p_new)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
        
        else:
            # It's not in the set. Try to add it.
            temp_coords = np.array(current_set)
            if check_point_validity(p_rand, temp_coords):
                # Accept
                current_set.append(p_rand)
                current_set_set.add(p_rand)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
        
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature

    return best_set

# Expose the refined version as the main construct
construct = construct_refined
