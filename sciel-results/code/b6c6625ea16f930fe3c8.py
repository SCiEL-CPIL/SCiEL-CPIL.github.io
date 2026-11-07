import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and bitmask
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        edge_set.add((v, u)) # Store both for easy lookup? 
                             # Actually, for biclique [L, R], we need (u,v) for u in L, v in R.
                             # If we store adj[u] as bitmask of v, we can check (1<<v) & adj[u].

    # Seed Solution
    seed_bicliques = []
    for u in range(n):
        # Get neighbors v such that (u, v) is in edges and u < v?
        # The seed logic: [v for a,v in P['edges'] if a==u]
        # This implies P['edges'] has specific orientation.
        # Let's replicate seed exactly to get baseline.
        right_side = [v for a, v in edges if a == u]
        if right_side:
            seed_bicliques.append([[u], right_side])
            
    best_bicliques = seed_bicliques
    best_count = len(seed_bicliques)

    # Greedy Optimization
    # We try to find a partition with fewer bicliques.
    # State: remaining_edges_mask[u] is bitmask of v such that (u,v) is still to be covered.
    # Note: If we cover (u,v), we must remove it from both u's and v's perspective?
    # Yes, because it's a single edge in the graph.
    
    # To speed up, we only iterate on u < v edges to avoid double counting.
    # But bitmask operations are on the full adjacency.
    # When we form a biclique L, R:
    # For u in L, v in R:
    #   The edge (u,v) must be present.
    #   We remove (u,v) from remaining.
    
    def try_greedy(restarts=50):
        nonlocal best_bicliques, best_count
        for _ in range(restarts):
            if time.time() > deadline:
                return
            
            # Initialize remaining masks
            rem_mask = adj[:]  # Copy
            # We need to ensure we only consider edges (u,v) with u < v to count them once?
            # No, the partition is on the set of edges.
            # If we use undirected edges, (u,v) is one edge.
            # In rem_mask, if we clear bit v in rem_mask[u], we should also clear bit u in rem_mask[v].
            
            edges_left = len(edges)
            current_bicliques = []
            
            while edges_left > 0:
                if time.time() > deadline:
                    return
                
                best_local = None
                best_size = 0
                
                # Try a few random starting edges
                # Pick random u, v such that rem_mask[u] has bit v
                candidates = []
                for u in range(n):
                    if rem_mask[u]:
                        # Pick a few random v from rem_mask[u]
                        mask = rem_mask[u]
                        # Get list of set bits
                        vs = []
                        m = mask
                        while m:
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            vs.append(v)
                            m ^= lsb
                        # Shuffle and take first few
                        random.shuffle(vs)
                        for v in vs[:5]:
                            candidates.append((u, v))
                
                if not candidates:
                    break # Should not happen if edges_left > 0
                
                random.shuffle(candidates)
                
                for u, v in candidates[:20]: # Limit attempts per step
                    if time.time() > deadline:
                        return
                    # Ensure edge is still there
                    if not (rem_mask[u] & (1 << v)):
                        continue
                    
                    # Expand biclique
                    L = {u}
                    R = {v}
                    
                    # Iterative expansion
                    changed = True
                    while changed:
                        changed = False
                        # Expand L: find u' not in L such that u' connected to all R in rem_mask
                        # Mask for R:
                        mask_R = 0
                        for rv in R:
                            mask_R |= (1 << rv)
                        
                        # Candidates for L: all u' such that (rem_mask[u'] & mask_R) == mask_R
                        # And u' not in L
                        # And u' != v? No, u' can be anything not in L.
                        # But we must ensure we don't add u' if the edges (u', rv) are not in rem_mask.
                        # The condition (rem_mask[u'] & mask_R) == mask_R ensures that for all rv in R, (u', rv) is in rem_mask.
                        
                        # Optimization: Iterate u' from 0 to n-1
                        for u_prime in range(n):
                            if u_prime in L:
                                continue
                            if (rem_mask[u_prime] & mask_R) == mask_R and mask_R != 0:
                                # Check if u_prime actually has edges to R in rem_mask
                                # If mask_R is 0, condition is true, but R is not empty.
                                L.add(u_prime)
                                changed = True
                                
                        # Expand R: find v' not in R such that v' connected to all L in rem_mask
                        mask_L = 0
                        for lu in L:
                            mask_L |= (1 << lu)
                        
                        for v_prime in range(n):
                            if v_prime in R:
                                continue
                            # Check edges (lu, v_prime) in rem_mask for all lu in L
                            # This is equivalent to: for all lu in L, (rem_mask[lu] & (1<<v_prime)) != 0
                            # Or: (rem_mask[lu] & mask_v_prime) == mask_v_prime ? No.
                            # We need: For each lu, bit v_prime is set in rem_mask[lu].
                            # Let's check:
                            valid = True
                            for lu in L:
                                if not (rem_mask[lu] & (1 << v_prime)):
                                    valid = False
                                    break
                            if valid:
                                R.add(v_prime)
                                changed = True

                    # Calculate size
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_local = (L, R)

                if not best_local:
                    # Fallback: pick single edge
                    # Find any remaining edge
                    found = False
                    for u in range(n):
                        if rem_mask[u]:
                            m = rem_mask[u]
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            if (rem_mask[v] & (1 << u)): # Ensure undirected consistency in rem_mask?
                                # We manage rem_mask symmetrically, so this should be true.
                                best_local = ({u}, {v})
                                best_size = 1
                                found = True
                                break
                    if not found:
                        break
                        
                L, R = best_local
                current_bicliques.append([list(L), list(R)])
                
                # Update rem_mask
                for u in L:
                    mask_R = 0
                    for v in R:
                        mask_R |= (1 << v)
                    rem_mask[u] &= ~mask_R
                    
                for v in R:
                    mask_L = 0
                    for u in L:
                        mask_L |= (1 << u)
                    rem_mask[v] &= ~mask_L
                    
                edges_left -= best_size
                
                # Sanity check: if edges_left < 0, something went wrong with size counting
                # best_size is len(L)*len(R). We removed exactly those edges.

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                # If we hit 1 biclique, we can stop
                if best_count == 1:
                    return

    try_greedy()

    # Return best found
    # Convert to required format: list of [left, right]
    # Ensure types are int and lists
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result
