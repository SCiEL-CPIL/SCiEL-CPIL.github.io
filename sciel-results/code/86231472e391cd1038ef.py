import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    # 1. Generate candidate pool
    # We focus on vectors with small squared norms (1, 2, 3) because the constraint
    # min_dist_sq >= max_norm_sq implies that if we pick a vector with large norm,
    # all other vectors must be very far away, limiting packing density.
    # Vectors with norm 1: ±e_i (22 vectors)
    # Vectors with norm 2: ±e_i ± e_j (220 vectors)
    # Vectors with norm 3: ±e_i ± e_j ± e_k (1320 vectors)
    # Total ~ 1562 vectors. This is small enough for exact MIS.
    
    candidates = []
    
    # Norm 1
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
        candidates.append(-v)
        
    # Norm 2
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Norm 3
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
                            
    # Remove duplicates and zero vector
    unique_candidates = []
    seen = set()
    for v in candidates:
        if np.all(v == 0):
            continue
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
            
    n_cand = len(unique_candidates)
    cand_array = np.array(unique_candidates, dtype=np.int64)
    
    # 2. Build Conflict Graph
    # Two vertices i and j are connected (conflict) if they cannot be in the same set.
    # Condition for valid set S: for all u, v in S, dist_sq(u, v) >= max_norm_sq(S).
    # Let M = max_norm_sq(S).
    # This condition is global, not just pairwise. However, we can approximate or handle it in the search.
    # A simpler necessary condition: If we fix a maximum norm N_max in the set, then for any pair u,v in S,
    # dist_sq(u,v) >= N_max.
    # Since N_max is determined by the "heaviest" vector in S, we can iterate over possible N_max values?
    # Or better: In the MIS branch and bound, we can maintain the current max norm.
    # But standard MIS assumes static conflicts.
    # Let's refine the conflict definition:
    # If we include vector v, the max norm becomes max(current_max, ||v||^2).
    # This makes the conflict dynamic.
    
    # Alternative Strategy:
    # The constraint is min_{u,v in S} ||u-v||^2 >= max_{w in S} ||w||^2.
    # Let R = max_{w in S} ||w||^2.
    # Then for all u,v in S, ||u-v||^2 >= R.
    # Also for all u in S, ||u||^2 <= R.
    # So S is a subset of {v : ||v||^2 <= R} such that pairwise distances are >= R.
    # We can iterate over possible R values (1, 2, 3, ...).
    # For a fixed R, let C_R = {v in candidates : ||v||^2 <= R}.
    # We want the maximum size subset S of C_R such that for all u,v in S, ||u-v||^2 >= R.
    # This IS a standard MIS problem on the graph G_R where edges connect u,v if ||u-v||^2 < R.
    # We solve MIS for each R, take the best.
    
    # What R values to consider?
    # Max norm in our pool is 3. So R can be 1, 2, 3.
    # If R=1: Only norm 1 vectors allowed. Dist must be >= 1.
    # If R=2: Norm 1,2 vectors allowed. Dist must be >= 2.
    # If R=3: Norm 1,2,3 vectors allowed. Dist must be >= 3.
    
    best_set = []
    best_size = 0
    
    for R in [1, 2, 3]:
        if time.time() > deadline:
            break
            
        # Filter candidates with norm_sq <= R
        norms_sq = np.sum(cand_array**2, axis=1)
        valid_indices = np.where(norms_sq <= R)[0]
        
        if len(valid_indices) < 2:
            continue
            
        sub_cands = cand_array[valid_indices]
        n_sub = len(sub_cands)
        
        # Build adjacency matrix for conflict graph G_R
        # Edge (i, j) exists if dist_sq(sub_cands[i], sub_cands[j]) < R
        # Note: i != j. Dist is always > 0 for distinct points.
        
        # We will use a bitset-based MIS solver.
        # Since n_sub is at most ~1562, we can use Python integers as bitsets.
        # But 1562 bits is large for simple Python int ops? No, Python ints are arbitrary precision.
        # However, constructing the full adjacency bitset for 1562 nodes might be slow if done naively.
        # Let's try to optimize.
        
        # Actually, for R=3, n_sub is ~1562.
        # For R=2, n_sub is ~242.
        # For R=1, n_sub is 22.
        
        # Let's solve MIS for each R.
        
        # Precompute norms for sub_cands
        sub_norms = norms_sq[valid_indices]
        
        # Conflict check: dist_sq(u, v) < R
        # dist_sq = ||u||^2 + ||v||^2 - 2 u.v
        # We need this to be < R to be a conflict.
        
        # Build adjacency list or bitsets
        # For speed, let's use a list of neighbors for each node, then convert to bitsets if needed.
        # Or just use a simple recursive backtracking with pruning.
        
        # Given the small size for R=1,2, and moderate for R=3, a well-optimized backtracking should work.
        
        # Let's implement a standard MIS branch and bound with degree ordering.
        
        # 1. Compute degrees
        # We need to know conflicts.
        # Let's compute the conflict matrix (boolean) or just check on the fly?
        # On the fly is O(N^2) per node check. With N=1500, N^2 = 2.25M.
        # If we do this 1500 times, it's 3.3B ops. Too slow in Python.
        
        # We must precompute conflicts.
        # Let's create an array `conflicts` of shape (n_sub, n_sub) boolean?
        # 1500x1500 = 2.25M booleans. This is fine memory-wise.
        # But computing it takes time.
        
        # dist_sq matrix calculation:
        # D = ||U||^2 + ||V||^2 - 2 U.V
        # U.V can be computed as matrix multiply.
        
        U = sub_cands
        # norms_col = sub_norms.reshape(-1, 1)
        # norms_row = sub_norms.reshape(1, -1)
        # dots = U @ U.T
        # dist_sq = sub_norms[:, None] + sub_norms[None, :] - 2 * dots
        
        # This is fast with NumPy.
        
        if time.time() > deadline:
            break
            
        dots = U @ U.T
        dist_sq = sub_norms[:, None] + sub_norms[None, :] - 2 * dots
        
        # Conflict if dist_sq < R
        # Note: dist_sq[i,i] = 0. We don't care about self-loops.
        # For i != j, if dist_sq[i,j] < R, then i and j conflict.
        
        # We want the Max Independent Set.
        # Independent Set means NO edges between selected nodes.
        # So we want a set S such that for all i,j in S, dist_sq[i,j] >= R.
        
        # Let's build the "compatibility" graph where edge means "can be together".
        # Compat[i,j] = (dist_sq[i,j] >= R)
        
        # We will use a branch and bound algorithm on the incompatibility graph (conflicts).
        # Or on the compatibility graph?
        # Standard MIS is on the graph where edges are conflicts.
        
        # Let's use a simple recursive search with pruning.
        # Order nodes by degree (number of conflicts) descending?
        # Actually, for MIS, picking high degree nodes first for pruning is often good?
        # Or low degree?
        # Let's stick to a known heuristic: Order by degree (conflict count) descending.
        
        # Precompute conflict bitsets for each node?
        # Python ints can hold bitsets.
        # Let's create `conflict_bits[i]` as an integer where bit j is 1 if i conflicts with j.
        
        conflict_bits = [0] * n_sub
        # We only need upper triangle for computation, but bitsets are symmetric.
        # Loop i from 0 to n_sub-1
        # Loop j from i+1 to n_sub-1
        # if dist_sq[i,j] < R:
        #    conflict_bits[i] |= (1 << j)
        #    conflict_bits[j] |= (1 << i)
        
        # This double loop is O(N^2) in Python. 1500^2 / 2 = 1.1M iterations.
        # This is acceptable (< 1s).
        
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                if dist_sq[i, j] < R:
                    conflict_bits[i] |= (1 << j)
                    conflict_bits[j] |= (1 << i)
                    
        # Now solve MIS on this graph.
        # We want max size subset with no conflicts.
        
        # Algorithm: Branch and Bound
        # State: current set, available candidates (bitmask)
        # Pruning: if len(current) + popcount(available) <= best_size, prune.
        
        best_size_R = 0
        best_set_R = []
        
        # Initial available mask: all 1s for n_sub bits
        full_mask = (1 << n_sub) - 1
        
        # We need a fast popcount. Python 3.8+ has int.bit_count().
        
        # Recursive function
        def solve_mis(available_mask, current_set, current_size):
            nonlocal best_size_R, best_set_R
            
            if time.time() > deadline:
                return
                
            if available_mask == 0:
                if current_size > best_size_R:
                    best_size_R = current_size
                    best_set_R = current_set[:]
                return
                
            # Prune
            if current_size + available_mask.bit_count() <= best_size_R:
                return
                
            # Select a pivot node
            # Heuristic: Choose node with max degree in the induced subgraph?
            # Or just pick the first available node?
            # Let's pick the node with the highest number of conflicts within the available set.
            # This helps prune.
            
            # To find max degree, we need to count bits in (conflict_bits[i] & available_mask) for each i in available.
            # This is O(N) per step.
            
            # Optimization: Just pick the first available node?
            # For N=1500, this might be slow if branching factor is high.
            # Let's try picking the node with max conflict count in available set.
            
            max_deg = -1
            pivot = -1
            temp_mask = available_mask
            while temp_mask:
                # Get lowest set bit index
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                temp_mask ^= lsb
                
                deg = (conflict_bits[idx] & available_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = idx
                    
            if pivot == -1:
                return
                
            # Branch 1: Include pivot
            # New available: available_mask & ~conflict_bits[pivot] & ~(1 << pivot)
            # Actually, if we include pivot, we remove pivot and its neighbors.
            new_mask = available_mask & ~(conflict_bits[pivot] | (1 << pivot))
            current_set.append(pivot)
            solve_mis(new_mask, current_set, current_size + 1)
            current_set.pop()
            
            # Branch 2: Exclude pivot
            # New available: available_mask & ~(1 << pivot)
            new_mask = available_mask & ~(1 << pivot)
            solve_mis(new_mask, current_set, current_size)
            
        # Start search
        # To improve speed, we can initialize best_size_R with a greedy solution.
        # Greedy: Pick node with min degree, remove it and neighbors, repeat.
        greedy_set = []
        greedy_mask = full_mask
        while greedy_mask:
            # Pick node with min degree in greedy_mask
            min_deg = float('inf')
            min_node = -1
            temp_mask = greedy_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                temp_mask ^= lsb
                deg = (conflict_bits[idx] & greedy_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_node = idx
            if min_node == -1: break
            greedy_set.append(min_node)
            greedy_mask &= ~(conflict_bits[min_node] | (1 << min_node))
            
        best_size_R = len(greedy_set)
        best_set_R = greedy_set[:]
        
        # Now run exact search
        solve_mis(full_mask, [], 0)
        
        if best_size_R > best_size:
            best_size = best_size_R
            # Map indices back to original vectors
            best_set = [sub_cands[i] for i in best_set_R]
            
    # Final verification
    if len(best_set) >= 2:
        result = np.array(best_set, dtype=np.int64)
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return result
        else:
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    else:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
