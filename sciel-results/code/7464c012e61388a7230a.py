import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Strategy:
    # 1. Generate all integer vectors with squared norm 4 (weight 4, entries +/-1) and norm 2 (weight 2, entries +/-1).
    #    Total candidates: C(11,4)*16 + C(11,2)*4 = 330*16 + 55*4 = 5280 + 220 = 5500.
    # 2. The condition for a valid set is: min_pair_dist_sq >= max_norm_sq.
    #    If we mix norms, max_norm_sq = 4. So we need dist_sq >= 4 for all pairs.
    #    dist_sq = |u|^2 + |v|^2 - 2u.v
    #    Case 1: |u|^2=4, |v|^2=4 => 8 - 2u.v >= 4 => u.v <= 2.
    #    Case 2: |u|^2=4, |v|^2=2 => 6 - 2u.v >= 4 => u.v <= 1.
    #    Case 3: |u|^2=2, |v|^2=2 => 4 - 2u.v >= 4 => u.v <= 0.
    #
    #    This defines a conflict graph where an edge exists if the condition is violated.
    #    We want a Maximum Independent Set (MIS) in this graph.
    #
    #    Since 5500 nodes is too many for exact MIS, we use a greedy approach with random restarts,
    #    but optimized with bitsets for speed.
    
    candidates = []
    # Norm 2
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates.append(v)
            
    # Norm 4
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates.append(v)
            
    V = np.array(candidates, dtype=np.int64)
    N = len(V)
    
    # Compute norms squared
    norms_sq = np.sum(V * V, axis=1)
    
    # Compute dot product matrix in chunks to avoid memory blowup if N was larger, but 5500x5500 is ~24MB, fine.
    # However, building the conflict graph adjacency list is the key step.
    
    # We will store the conflict graph as a list of bitsets (Python integers) for fast intersection/union.
    # N=5500 requires ~688 64-bit words, or just use Python big ints which are efficient for bitwise ops.
    
    conflict_bitsets = [0] * N
    
    # Compute dot products row by row
    for i in range(N):
        if time.time() > deadline - 30:
             # Fallback to a simple valid set if we run out of time building graph
             # Return just norm 2 vectors? Or best found so far?
             # Since we haven't started search, return a small valid set.
             return np.array(candidates[:55], dtype=np.int64)
        
        dots_row = V[i] @ V.T
        # Determine conflicts for node i
        # Condition: dist_sq < max_norm_sq
        # dist_sq = norms_sq[i] + norms_sq[j] - 2*dots_row[j]
        # max_norm_sq = 4 (since we have norm 4 vectors)
        # So conflict if norms_sq[i] + norms_sq[j] - 2*dots_row[j] < 4
        # => 2*dots_row[j] > norms_sq[i] + norms_sq[j] - 4
        
        # Let's compute the threshold for each j
        # Actually, simpler:
        # If norms_sq[i] == 4 and norms_sq[j] == 4: conflict if dots_row[j] > 2
        # If norms_sq[i] == 4 and norms_sq[j] == 2: conflict if dots_row[j] > 1
        # If norms_sq[i] == 2 and norms_sq[j] == 4: conflict if dots_row[j] > 1
        # If norms_sq[i] == 2 and norms_sq[j] == 2: conflict if dots_row[j] > 0
        
        # We can vectorize this logic.
        # Create a mask of valid j's (j != i)
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        
        # Calculate required dot product threshold
        # Conflict if dots_row[j] > (norms_sq[i] + norms_sq[j] - 4) / 2
        # Since dots are integers, we can check:
        # 2 * dots_row[j] > norms_sq[i] + norms_sq[j] - 4
        
        left_side = 2 * dots_row
        right_side = norms_sq[i] + norms_sq - 4
        
        conflict_mask = left_side > right_side
        conflict_mask[i] = False # Self is not a conflict in the independent set sense (we just don't pick it twice)
        
        # Convert to bitset
        # We only care about bits set where conflict_mask is True
        # To create a big int bitset from a boolean array:
        # We can use np.packbits or just iterate if N is small, but N=5500.
        # Using Python's int from bytes is fast.
        # But we need to be careful with endianness.
        # Alternative: Just store as a set of indices?
        # Sets are slower for intersection than bitsets, but N=5500 might be okay for greedy?
        # Greedy step:
        #   Pick v
        #   Remove v and neighbors(v) from active
        #   If we use sets: active = active - neighbors[v]
        #   Intersection of sets of size ~100-200 is fast.
        #   Number of neighbors:
        #     Norm 4 vs Norm 4: dot > 2. Max dot is 4 (same vector).
        #     How many vectors have dot > 2?
        #     If u has weight 4, v has weight 4.
        #     dot = sum u_k v_k.
        #     If they share 4 positions: dot = 4 (if signs match), 2 (2 match 2 diff), 0 (1 match 3 diff), -2, -4.
        #     So dot > 2 means dot=4. Only the vector itself.
        #     Wait, if they share 3 positions?
        #     u: 1,1,1,1,0... v: 1,1,1,0,1...
        #     dot = 3. 3 > 2. Conflict.
        #     So many conflicts.
        
        # Let's stick to bitsets for speed.
        # To create bitset:
        # indices = np.where(conflict_mask)[0]
        # bitset = 0
        # for idx in indices: bitset |= (1 << idx)
        # This loop is slow in Python.
        
        # Better: use numpy to pack into bytes and convert to int.
        # But we need to handle the bit order.
        # Let's try a different approach: Store neighbors as a list of arrays?
        # And use a boolean mask for active set.
        # active_mask = np.ones(N, dtype=bool)
        # active_mask[neighbors] = False
        # This is O(k) where k is number of neighbors.
        # Total work per greedy run: Sum of degrees of selected nodes.
        # If we select ~500 nodes, and each has ~100 neighbors, that's 50,000 operations.
        # With 590s, we can do many restarts.
        
        # Let's go with the boolean mask approach, it's simpler and likely fast enough.
        # We need to precompute neighbors as lists of indices.
        
        neighbors_list = np.where(conflict_mask)[0]
        conflict_bitsets[i] = neighbors_list # Storing as array for now

    # Greedy Search
    best_indices = []
    best_k = 0
    
    # Initial best: just all norm 2 vectors?
    # Norm 2 vectors have dot <= 0 with each other?
    # u, v weight 2.
    # If disjoint: dot=0. dist^2 = 4. OK.
    # If share 1: dot=1 or -1.
    # If dot=1: dist^2 = 2+2-2=2 < 4. Conflict!
    # So we can't just take all norm 2 vectors.
    # We need to find an independent set among them too.
    
    while time.time() < deadline - 5:
        order = np.random.permutation(N)
        active_mask = np.ones(N, dtype=bool)
        current_indices = []
        
        for v in order:
            if not active_mask[v]:
                continue
            current_indices.append(v)
            active_mask[v] = False
            # Remove neighbors
            if len(conflict_bitsets[v]) > 0:
                active_mask[conflict_bitsets[v]] = False
                
        k = len(current_indices)
        if k > best_k:
            best_k = k
            best_indices = current_indices
            
            # Optional: Local improvement?
            # Try to add more? No, it's maximal.
            # Try to swap?
            # For now, just restart.
            
            # If we get a very good result, maybe we can stop?
            # Theoretical max is likely around 1000-2000?
            # Current best known for integer kissing number 11 is likely small.
            # Let's just run.

    if best_k == 0:
        # Fallback
        return np.array(candidates[:55], dtype=np.int64)
        
    result = V[best_indices].astype(np.int64)
    return result
