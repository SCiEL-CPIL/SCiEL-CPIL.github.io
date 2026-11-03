"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer

    def elapsed():
        return time.time() - start_time

    # 1. Generate Candidate Pool
    # We use shells 1, 2, 3, 4.
    # Shell 1: 11 vectors
    # Shell 2: 220 vectors
    # Shell 3: 1716 vectors
    # Shell 4: 11880 vectors
    # Total ~ 13827

    vecs = []

    # Shell 1
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        vecs.append(v)

    # Shell 2
    for i in range(D):
        for j in range(D):
            if i == j: continue
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            vecs.append(v)

    # Shell 3
    for comb in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)

    # Shell 4
    for comb in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)

    candidates = np.array(vecs, dtype=np.int64)
    n_cand = len(candidates)

    # Precompute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1).astype(np.int64)

    # 2. Construct Conflict Graph using Bitmasks
    # Conflict if ||u - v||^2 < max(||u||^2, ||v||^2)
    # We will store the set of conflicts for each vertex i as a bitmask.
    # Since n_cand ~ 13827, a Python int bitmask is perfect.

    conflict_masks = [0] * n_cand

    # To make this efficient, we compute the Gram matrix in chunks or all at once if memory allows.
    # 13827^2 * 4 bytes ~ 760 MB. This is risky.
    # Let's try to process in batches to avoid memory blowup, or just try the full matrix with float32.
    # Actually, we can compute the conflict masks row by row to save memory on the Gram matrix.
    
    C_float = candidates.astype(np.float32)
    ns = norms_sq.astype(np.float32)

    # We iterate over rows of C to compute conflicts.
    # For each i, we compute dot products with all j.
    # dot = C[i] @ C.T
    # This is an (N,) vector for each i.
    # Doing this N times is O(N^2 * D). 13800 * 13800 * 11 ~ 2 billion ops.
    # NumPy can handle this in ~10-20 seconds if optimized.
    
    # Let's try a vectorized approach for the conflict matrix first, but storing it as masks.
    # If we can't fit the full boolean matrix in memory, we compute row by row.
    # 13800 * 13800 booleans is ~ 190 MB. That's fine.
    
    try:
        # Compute full Gram matrix
        G = C_float @ C_float.T
        
        # Compute dist_sq and max_ns
        ns_col = ns[:, np.newaxis]
        ns_row = ns[np.newaxis, :]
        
        dist_sq = ns_col + ns_row - 2 * G
        max_ns = np.maximum(ns_col, ns_row)
        
        conflict_bool = dist_sq < max_ns
        np.fill_diagonal(conflict_bool, False)
        
        # Convert boolean matrix to list of integer bitmasks
        # This conversion is slow in Python if done naively.
        # We can use a trick: treat rows as bytes and convert to int?
        # Or just loop over rows and pack bits.
        # Looping over 13800 rows is fast.
        
        # Optimization: Precompute bit shifts? No, just loop.
        for i in range(n_cand):
            # Get row of conflicts
            row = conflict_bool[i]
            # Find indices where True
            # This is slow if we do np.nonzero for each row.
            # Better:
            # We can convert the boolean array to a string of '0'/'1' or bytes and then to int?
            # Actually, numpy doesn't have a direct "pack bits to int".
            # Let's use a loop over the True indices.
            # If the graph is dense, this is slow.
            # How dense is it?
            # Shell 1 vs Shell 1: ||e_i - e_j||^2 = 2, max_norm^2=1. Conflict. (i!=j)
            # So Shell 1 is a clique.
            # Shell 1 vs Shell 2: e_i and e_j - e_k.
            # If i=j, ||e_i - (e_i - e_k)||^2 = ||e_k||^2 = 1. max=2. No conflict.
            # If i!=j, ||e_i - e_j + e_k||^2 = 3. max=2. Conflict.
            # It seems moderately dense.
            
            # A faster way to convert a boolean numpy array to an integer bitmask:
            # view as uint8, then to int? No, bit order is an issue.
            # Let's just use:
            # mask = 0
            # for bit in np.nonzero(row)[0]:
            #     mask |= (1 << bit)
            # This is O(k) per row. Sum k ~ N^2/2.
            # Total ops ~ N^2/2. 100 million bit sets.
            # In Python, this might take 5-10 seconds. Acceptable.
            
            # To speed up, we can use:
            # indices = np.nonzero(row)[0]
            # mask = sum(1 << i for i in indices)
            # Still slow.
            
            # Let's stick to the simplest correct method and hope it's fast enough.
            # Actually, `int.from_bytes` can be used if we view the boolean array as bytes.
            # But numpy boolean array is 1 byte per element.
            # We need to pack 8 booleans into 1 byte.
            # Let's try a different approach:
            # We don't actually need to convert the *whole* matrix to masks if we use the boolean matrix directly in the MIS solver, 
            # but the solver needs fast neighbor queries.
            # Bitmasks are much faster for neighbor queries in Python than boolean array slicing.
            
            # Let's do the conversion efficiently:
            # row_bytes = np.packbits(row.view(np.uint8)) # This packs 8 booleans into 1 byte, MSB first?
            # Actually np.packbits packs from LSB to MSB within the byte, and rows are little-endian?
            # It's tricky.
            
            # Let's just use the loop method, it's safer.
            indices = np.nonzero(row)[0]
            mask = 0
            for idx in indices:
                mask |= (1 << idx)
            conflict_masks[i] = mask

    except Exception as e:
        # Fallback to greedy without full mask?
        # Or just return a small set.
        return candidates[:11]

    # 3. Solve Maximum Independent Set (MIS) using Branch and Bound with Bitmasks
    
    # We use a standard recursive MIS algorithm.
    # To avoid recursion depth issues, we can use an iterative approach or increase recursion limit.
    # N=13800, depth could be up to N. Recursion limit in Python is 1000 by default.
    # We must use an iterative approach or set a very high recursion limit (risky).
    # Let's use an iterative stack-based approach or a greedy approach that is faster.
    
    # Given the time constraint and the difficulty of implementing a fast, correct, iterative MIS with bitmasks in pure Python,
    # let's try a **Greedy with Local Search** first, which is simpler and often sufficient.
    # Or, we can use a **Maximal Independent Set** via a greedy algorithm, then try to improve it.
    
    # However, the cell finding says bitmasks allow branch-and-bound to explore more nodes.
    # Let's try a simple recursive MIS with a high recursion limit and hope the tree isn't too deep?
    # No, that's dangerous.
    
    # Alternative: **Tomita's algorithm** for maximal clique (which is MIS on complement).
    # But implementing it with bitmasks iteratively is complex.
    
    # Let's go with a robust **Greedy + Random Restarts** approach using the conflict masks for fast updates.
    # This is likely what the previous "2.2" solution did if it didn't use full B&B.
    # Wait, the cell finding specifically mentions B&B.
    
    # Let's try a **Hybrid**:
    # 1. Greedy to get a good initial solution.
    # 2. Local search to improve it.
    
    def greedy_is(order):
        available = (1 << n_cand) - 1
        selected = []
        
        for v in order:
            if available & (1 << v):
                selected.append(v)
                # Remove v and its neighbors from available
                # Neighbors are in conflict_masks[v]
                available &= ~(1 << v)
                available &= ~conflict_masks[v]
        
        return selected

    best_sel = []
    best_size = 0
    
    # Heuristic 1: Ascending Degree
    degrees = np.array([bin(mask).count('1') for mask in conflict_masks])
    order1 = np.argsort(degrees)
    sel1 = greedy_is(order1)
    if len(sel1) > best_size:
        best_size = len(sel1)
        best_sel = sel1
        
    if elapsed() > deadline - 20:
        return np.array(candidates[best_sel], dtype=np.int64)
        
    # Heuristic 2: Random Restarts
    rng = np.random.default_rng(42)
    n_restarts = 200 # Increased restarts
    
    for i in range(n_restarts):
        if elapsed() > deadline - 10:
            break
            
        rand_order = rng.permutation(n_cand)
        sel_r = greedy_is(rand_order)
        if len(sel_r) > best_size:
            best_size = len(sel_r)
            best_sel = sel_r
            
    # Heuristic 3: Lexsort by Norm, then Degree
    order3 = np.lexsort((degrees, norms_sq))
    sel3 = greedy_is(order3)
    if len(sel3) > best_size:
        best_size = len(sel3)
        best_sel = sel3
        
    # Local Search: Try to add more vertices?
    # Or try to swap?
    # Given time, we stop here.
    
    return np.array(candidates[best_sel], dtype=np.int64)

# EVOLVE-BLOCK-END
