import numpy as np
import time
import itertools
import random

# Precompute all masks and popcounts
N = 18
TOTAL_MASKS = 1 << N
# all_masks = np.arange(TOTAL_MASKS, dtype=np.int64)
# popcounts = np.array([bin(i).count('1') for i in range(TOTAL_MASKS)], dtype=np.int32)

# To save memory and time, we might not need all masks at once if we search by size.
# But vectorized check on all masks is fast enough.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Precompute masks and popcounts
    all_masks = np.arange(TOTAL_MASKS, dtype=np.int64)
    # Efficient popcount for numpy array
    # Using a lookup table or bit tricks
    # bin(i).count('1') is slow. 
    # np.unpackbits works on bytes.
    # Let's use a simple loop for 2^18 items, it's fast enough once.
    popcounts = np.zeros(TOTAL_MASKS, dtype=np.int32)
    for i in range(1, TOTAL_MASKS):
        popcounts[i] = popcounts[i >> 1] + (i & 1)

    def get_triangles(adj):
        """
        adj: list of int bitmasks for each vertex.
        Returns list of triangle bitmasks.
        """
        triangles = []
        for u in range(N):
            # Neighbors of u
            nu = adj[u]
            # For each pair of neighbors v, w of u, if (v,w) is edge, then (u,v,w) is triangle
            # To avoid duplicates, only consider v < w
            # Iterate bits in nu
            # We can iterate v in range(u+1, N) if adj[u] has bit v?
            # Better: iterate all v, w in nu with v < w
            # Since N=18, nu has <= 17 bits.
            # Pairs: ~136 max.
            neighbors = [i for i in range(N) if (nu >> i) & 1]
            for i in range(len(neighbors)):
                v = neighbors[i]
                for j in range(i+1, len(neighbors)):
                    w = neighbors[j]
                    # Check if v-w is edge
                    if (adj[v] >> w) & 1:
                        # Triangle u, v, w
                        mask = (1 << u) | (1 << v) | (1 << w)
                        triangles.append(mask)
        return triangles

    def is_k4_free(adj):
        """
        Check if graph with adjacency bitmasks adj is K4-free.
        """
        for u in range(N):
            nu = adj[u]
            # Common neighbors of pairs in nu?
            # Actually, K4 exists if there is an edge (v,w) in nu such that N(v) cap N(w) contains u and another?
            # Simpler: For every edge (v,w) in the graph, check if N(v) cap N(w) contains an edge?
            # No.
            # K4 is a clique of size 4.
            # Iterate all edges (u,v).
            # Common neighbors C = N(u) cap N(v).
            # If C contains an edge (w,x), then u,v,w,x form K4.
            for u in range(N):
                nu = adj[u]
                # Iterate v > u where edge exists
                for v in range(u+1, N):
                    if not ((nu >> v) & 1):
                        continue
                    # Common neighbors
                    c = nu & adj[v]
                    # Check if subgraph induced by c has any edge
                    # Iterate bits in c
                    # If any two bits in c are connected by an edge in G
                    # We can check: for each w in c, if (adj[w] & c) != 0 (excluding w itself? adj[w] doesn't have w)
                    # If (adj[w] & c) != 0, then w is connected to some x in c.
                    # Since we just need to know if ANY edge exists in induced subgraph of c.
                    # If c has 0 or 1 bits, no edge.
                    # If c has >= 2 bits:
                    #   for w in bits(c):
                    #       if (adj[w] & c) != 0: return False
                    # Optimization: 
                    # If c & (c-1) == 0, c is power of 2 or 0. No edge possible.
                    if c & (c - 1): # c has at least 2 bits
                        # Check edges in c
                        # We can just check if any w in c has neighbor in c
                        # To speed up, we can precompute? No, adj changes.
                        # But N=18 is small.
                        # Iterate bits in c
                        cc = c
                        while cc:
                            w_bit = cc & -cc
                            w = w_bit.bit_length() - 1
                            if (adj[w] & c):
                                return False
                            cc -= w_bit
        return True

    def max_triangle_free_size(adj):
        triangles = get_triangles(adj)
        if not triangles:
            return N
        T_np = np.array(triangles, dtype=np.int64)
        # Vectorized check
        # all_masks shape (2^18, 1), T_np shape (1, T)
        # This creates a large array. 
        # 2^18 * T. If T=200, 50M elements. 
        # It might be slow or memory heavy if called thousands of times.
        # Alternative: Iterate masks by popcount descending?
        # But vectorized is C-speed.
        # Let's try to be smarter.
        # We only care about the MAX size.
        # We can check masks of size k from N down to 1.
        # But generating masks of specific size is hard to vectorize simply.
        # Let's stick to vectorized all-mask check for now, it's likely fast enough for ~1000 calls.
        
        # To avoid huge memory allocation every time, we can reuse buffers?
        # Or just do it.
        # intersections = all_masks[:, None] & T_np[None, :]
        # contained = intersections == T_np[None, :]
        # has_tri = contained.any(axis=1)
        # valid = ~has_tri
        # max_size = np.max(popcounts[valid])
        
        # Optimization: If T is large, this is slow.
        # But in K4-free graphs, T is bounded.
        
        # Let's try a hybrid: If T is small, use vectorized.
        # Actually, 50M operations in NumPy is ~0.1s.
        # 1000 calls = 100s. Too slow.
        
        # We need a faster evaluator.
        # Branch and Bound in Python might be faster if the answer is small?
        # If answer is 10, we check masks of size 18..11.
        # Number of masks: ~2^18 / 2 = 130k.
        # Python loop over 130k masks, each checking T triangles.
        # If T=20, 130k * 20 = 2.6M ops. ~0.5s.
        # Still slow.
        
        # How to speed up triangle check?
        # Precompute `triangle_free_mask`?
        # We can compute `is_triangle_free[mask]` using DP?
        # `is_tf[0] = True`
        # `is_tf[mask] = is_tf[mask ^ lsb] and (no triangle formed by adding lsb)`
        # Triangle formed by adding v to S if N_S(v) has an edge.
        # We can precompute `has_edge_in_subset[mask]`?
        # `has_edge_in_subset[mask]` is True if subgraph induced by mask has an edge.
        # This can be computed for all 2^18 masks in ~0.5s using DP?
        # `has_edge[0] = False`
        # `has_edge[mask] = has_edge[mask ^ lsb] or (adj[lsb] & (mask ^ lsb)) != 0`
        # This is O(2^18) operations.
        # We can do this for the CURRENT graph in ~0.1-0.2s.
        # Then `is_triangle_free[mask]` is True if `mask` contains no triangle.
        # Wait, the condition "no triangle formed by adding v" is equivalent to "N_S(v) is independent".
        # This is NOT the same as "S is triangle-free".
        # If S is triangle-free, adding v is valid iff N_S(v) is independent.
        # So we can compute `valid_add[v][S]`? No.
        
        # Let's use the DP for `has_edge_in_subset`.
        # Let `E[mask]` be True if G[mask] has an edge.
        # `E[0] = False`
        # `E[mask] = E[mask without lsb] OR (adj[lsb] & (mask without lsb)) != 0`
        # This takes ~2^18 steps.
        
        # Now, how to find max triangle-free subset?
        # A subset S is triangle-free if it contains no triangle.
        # This is NOT directly related to E[S].
        
        # Let's go back to: `S` is triangle-free iff for all triangles t, S does not contain t.
        # We can precompute `contains_triangle[mask]`?
        # `CT[0] = False`
        # `CT[mask] = CT[mask without lsb] OR (lsb completes a triangle with mask without lsb)`
        # `lsb` completes a triangle with `S'` if there is an edge (u,v) in `S'` such that (u,lsb) and (v,lsb) are edges.
        # i.e., `N_{S'}(lsb)` has an edge.
        # So `CT[mask] = CT[S'] OR E[N_{S'}(lsb)]`
        # We have `E` precomputed for all masks!
        # So we can compute `CT` for all masks in O(2^18) time.
        
        # Algorithm:
        # 1. Compute `E` array (has edge in induced subgraph) for all 2^18 masks.
        # 2. Compute `CT` array (contains triangle) for all 2^18 masks.
        # 3. Find max popcount where `CT` is False.
        
        # Time for step 1 & 2: ~2 * 2^18 operations = 500k ops. Very fast in NumPy?
        # NumPy vectorization for DP is hard due to dependencies.
        # But we can do it in a Python loop? 262k iterations. ~0.1s.
        # Total eval time ~0.2s.
        # 1000 evals = 200s. Still too slow.
        
        # Can we do it in C? No.
        # Can we optimize the search to need fewer evals?
        # Yes, if we use a good heuristic.
        
        # Let's try to make the evaluator faster.
        # If we only need the MAX size, we can stop early?
        # In the DP for CT, we can't easily stop.
        
        # Alternative: Randomized Local Search with approximate evaluation?
        # No, score is exact.
        
        # Let's try to reduce the number of evaluations.
        # Start with K6,6,6 (score 12).
        # Try to find score 11 or 10.
        # If we find 10, we stop.
        
        # Let's implement the fast evaluator using the DP trick in Python, but optimized.
        
        def fast_eval(adj):
            # adj: list of int bitmasks
            # Step 1: Compute E (has edge)
            E = np.zeros(TOTAL_MASKS, dtype=bool)
            # E[0] is False
            # Loop i from 1 to 2^18-1
            # lsb = i & -i
            # v = lsb.bit_length() - 1
            # rest = i ^ lsb
            # E[i] = E[rest] or ((adj[v] & rest) != 0)
            
            # This loop is 262k iterations.
            # In Python, ~0.2s.
            
            # Step 2: Compute CT (contains triangle)
            CT = np.zeros(TOTAL_MASKS, dtype=bool)
            # CT[0] is False
            # CT[i] = CT[rest] or E[ adj[v] & rest ]  # N_rest(v) has edge?
            # Wait, N_rest(v) is the set of neighbors of v in rest.
            # We need to check if the subgraph induced by N_rest(v) has an edge.
            # That is E[ adj[v] & rest ].
            
            # So CT[i] = CT[rest] or E[ adj[v] & rest ]
            
            # Step 3: Find max popcount where CT is False.
            # valid = ~CT
            # max_size = np.max(popcounts[valid])
            
            # We can combine Step 1 and 2?
            # No, E is needed for CT.
            
            # Let's write this loop carefully.
            E_list = [False] * TOTAL_MASKS
            CT_list = [False] * TOTAL_MASKS
            
            # Precompute lsb_index and rest for all masks?
            # lsb_idx = [0]*TOTAL_MASKS
            # rest = [0]*TOTAL_MASKS
            # for i in range(1, TOTAL_MASKS):
            #    lsb = i & -i
            #    lsb_idx[i] = lsb.bit_length()-1
            #    rest[i] = i ^ lsb
            
            # This precomputation is done once.
            
            for i in range(1, TOTAL_MASKS):
                v = lsb_idx[i]
                r = rest[i]
                # E[i]
                if E_list[r]:
                    E_list[i] = True
                else:
                    # Check if v has neighbor in r
                    if (adj[v] & r) != 0:
                        E_list[i] = True
                    else:
                        E_list[i] = False
                
                # CT[i]
                if CT_list[r]:
                    CT_list[i] = True
                else:
                    # Check if N_r(v) has edge
                    # N_r(v) = adj[v] & r
                    nr_v = adj[v] & r
                    if E_list[nr_v]:
                        CT_list[i] = True
                    else:
                        CT_list[i] = False
            
            # Find max
            max_sz = 0
            # Iterate valid masks
            # We can just loop i and check
            # Or use numpy
            # CT_np = np.array(CT_list)
            # valid = ~CT_np
            # max_sz = np.max(popcounts[valid])
            
            # To avoid numpy conversion overhead, loop in Python?
            # 262k iterations.
            for i in range(TOTAL_MASKS):
                if not CT_list[i]:
                    if popcounts[i] > max_sz:
                        max_sz = popcounts[i]
                        # Early exit if max_sz == N? Unlikely.
            
            return max_sz

    # Precompute lsb_idx and rest
    lsb_idx = [0] * TOTAL_MASKS
    rest = [0] * TOTAL_MASKS
    for i in range(1, TOTAL_MASKS):
        lsb = i & -i
        lsb_idx[i] = lsb.bit_length() - 1
        rest[i] = i ^ lsb

    # Initial Graph: K6,6,6
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * N
    for i in range(6):
        for j in range(6, 12):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
        for j in range(12, 18):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
    for i in range(6, 12):
        for j in range(12, 18):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
            
    # Verify K4-free (should be)
    # Score 12
    
    best_score = fast_eval(adj)
    best_adj = adj[:]
    
    # Local Search
    # Moves: Add edge, Remove edge
    # Add edge (u,v): Check K4-free.
    # Remove edge (u,v): Always K4-free.
    
    current_adj = adj[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.99
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        # Pick random move
        if random.random() < 0.5:
            # Add edge
            u = random.randint(0, N-1)
            v = random.randint(u+1, N-1)
            if (current_adj[u] >> v) & 1:
                continue
            # Check K4
            c = current_adj[u] & current_adj[v]
            # Check if c has edge
            is_k4 = False
            cc = c
            while cc:
                w_bit = cc & -cc
                w = w_bit.bit_length() - 1
                if (current_adj[w] & c):
                    is_k4 = True
                    break
                cc -= w_bit
            if is_k4:
                continue
            
            # Apply
            current_adj[u] |= 1 << v
            current_adj[v] |= 1 << u
            new_score = fast_eval(current_adj)
            
            # Accept?
            if new_score < current_score or random.random() < np.exp((current_score - new_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
            else:
                # Revert
                current_adj[u] ^= 1 << v
                current_adj[v] ^= 1 << u
        else:
            # Remove edge
            u = random.randint(0, N-1)
            v = random.randint(u+1, N-1)
            if not ((current_adj[u] >> v) & 1):
                continue
            # Apply
            current_adj[u] ^= 1 << v
            current_adj[v] ^= 1 << u
            new_score = fast_eval(current_adj)
            
            if new_score < current_score or random.random() < np.exp((current_score - new_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
            else:
                # Revert
                current_adj[u] |= 1 << v
                current_adj[v] |= 1 << u
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Convert best_adj to edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# Global precomputation for speed
N = 18
TOTAL_MASKS = 1 << N
popcounts = np.zeros(TOTAL_MASKS, dtype=np.int32)
for i in range(1, TOTAL_MASKS):
    popcounts[i] = popcounts[i >> 1] + (i & 1)

lsb_idx = [0] * TOTAL_MASKS
rest = [0] * TOTAL_MASKS
for i in range(1, TOTAL_MASKS):
    lsb = i & -i
    lsb_idx[i] = lsb.bit_length() - 1
    rest[i] = i ^ lsb

# Note: The fast_eval function inside construct uses these globals.
# But defining them inside construct is safer for isolation, but slower.
# Let's move them to module level for speed.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def fast_eval(adj):
        E_list = [False] * TOTAL_MASKS
        CT_list = [False] * TOTAL_MASKS
        
        for i in range(1, TOTAL_MASKS):
            v = lsb_idx[i]
            r = rest[i]
            if E_list[r]:
                E_list[i] = True
            else:
                if (adj[v] & r) != 0:
                    E_list[i] = True
                else:
                    E_list[i] = False
            
            if CT_list[r]:
                CT_list[i] = True
            else:
                nr_v = adj[v] & r
                if E_list[nr_v]:
                    CT_list[i] = True
                else:
                    CT_list[i] = False
        
        max_sz = 0
        for i in range(TOTAL_MASKS):
            if not CT_list[i]:
                if popcounts[i] > max_sz:
                    max_sz = popcounts[i]
        return max_sz

    # Initial Graph: K6,6,6
    adj = [0] * N
    for i in range(6):
        for j in range(6, 12):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
        for j in range(12, 18):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
    for i in range(6, 12):
        for j in range(12, 18):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
            
    best_score = fast_eval(adj)
    best_adj = adj[:]
    
    current_adj = adj[:]
    current_score = best_score
    
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if random.random() < 0.5:
            u = random.randint(0, N-1)
            v = random.randint(u+1, N-1)
            if (current_adj[u] >> v) & 1:
                continue
            c = current_adj[u] & current_adj[v]
            is_k4 = False
            cc = c
            while cc:
                w_bit = cc & -cc
                w = w_bit.bit_length() - 1
                if (current_adj[w] & c):
                    is_k4 = True
                    break
                cc -= w_bit
            if is_k4:
                continue
            
            current_adj[u] |= 1 << v
            current_adj[v] |= 1 << u
            new_score = fast_eval(current_adj)
            
            if new_score < current_score or random.random() < np.exp((current_score - new_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
            else:
                current_adj[u] ^= 1 << v
                current_adj[v] ^= 1 << u
        else:
            u = random.randint(0, N-1)
            v = random.randint(u+1, N-1)
            if not ((current_adj[u] >> v) & 1):
                continue
            current_adj[u] ^= 1 << v
            current_adj[v] ^= 1 << u
            new_score = fast_eval(current_adj)
            
            if new_score < current_score or random.random() < np.exp((current_score - new_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
            else:
                current_adj[u] |= 1 << v
                current_adj[v] |= 1 << u
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
