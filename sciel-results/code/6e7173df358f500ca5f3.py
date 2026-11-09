"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask adjacency
    # adj_mask[i] is an integer where bit j is 1 if (i,j) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute list of edges for iteration
    edge_list = [(min(u,v), max(u,v)) for u,v in edges]
    all_edges_mask = 0 # Not directly useful as edges are pairs, not single bits unless we map edge index to bit.
    
    # Map edge to index for bitmask of remaining edges? 
    # Number of edges ~ 150. 150 bits fits in Python int.
    edge_to_idx = {}
    idx_to_edge = []
    for i, (u, v) in enumerate(edge_list):
        edge_to_idx[(u, v)] = i
        idx_to_edge.append((u, v))
        
    # remaining_mask: bit i is 1 if edge i is remaining
    remaining_mask = (1 << len(edge_list)) - 1
    
    bicliques = []
    
    def get_biclique_edges_mask(L, R):
        """Returns bitmask of edges covered by biclique (L, R)."""
        mask = 0
        for u in L:
            for v in R:
                if u == v: continue
                e = (min(u,v), max(u,v))
                if e in edge_to_idx:
                    mask |= (1 << edge_to_idx[e])
        return mask

    def expand_biclique(L, R, rem_mask):
        """
        Expand L and R to be a maximal biclique with respect to the original graph,
        but only adding vertices that contribute to covering edges in rem_mask.
        """
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Candidates for L: vertices not in L or R
            # A vertex w can be in L if w is adjacent to all v in R (in original graph)
            # AND (w, v) is in rem_mask for at least one v in R? 
            # Actually, if we add w, we cover edges (w, v) for all v in R.
            # We should add w if it increases the set of covered remaining edges.
            # Since we are maximizing coverage, we should add w if it covers *any* new remaining edge.
            
            # Efficient check: 
            # For w to be in L, w must be in intersection of neighbors of all v in R.
            # common_neighbors = intersection of adj_mask[v] for v in R.
            # But we also need to check if the edge (w, v) is in rem_mask.
            
            # Let's compute potential additions for L
            # For each w not in L, R:
            #   if all (w, v) in edge_set for v in R:
            #       new_edges = {(w, v) for v in R if (w, v) in rem_mask}
            #       if new_edges is not empty: add w
            
            # To speed up, precompute for current R the set of vertices adjacent to all R.
            if R:
                common_L_candidates = (1 << n) - 1
                for v in R:
                    common_L_candidates &= adj_mask[v]
            else:
                common_L_candidates = (1 << n) - 1
                
            # Mask out L and R
            mask_LR = 0
            for x in L: mask_LR |= (1 << x)
            for x in R: mask_LR |= (1 << x)
            common_L_candidates &= ~mask_LR
            
            # Iterate through bits of common_L_candidates
            c = common_L_candidates
            while c:
                # Get lowest set bit
                lb = c & -c
                w = lb.bit_length() - 1
                c ^= lb
                
                # Check if adding w covers any remaining edge
                # Edges are (w, v) for v in R
                # We need to check if any (w, v) is in rem_mask
                covers_new = False
                for v in R:
                    e = (min(w,v), max(w,v))
                    if e in edge_to_idx:
                        idx = edge_to_idx[e]
                        if (rem_mask >> idx) & 1:
                            covers_new = True
                            break
                            
                if covers_new:
                    L.add(w)
                    changed = True
                    # Optimization: If L changes, common_L_candidates for R doesn't change?
                    # Wait, adding to L doesn't change the condition for adding to L (which depends on R).
                    # But it changes the condition for adding to R.
                    # So we can continue checking other L candidates?
                    # Actually, if we add w to L, the set of vertices that can be added to L in the *next* iteration
                    # might change? No, the condition "adjacent to all R" is independent of L.
                    # So we can add all valid L candidates in one pass?
                    # Yes, because adding w1 to L doesn't prevent w2 from being added to L (as long as w2 is adj to all R).
                    # So we can add all such w.
            
            # Similarly for R
            if L:
                common_R_candidates = (1 << n) - 1
                for u in L:
                    common_R_candidates &= adj_mask[u]
            else:
                common_R_candidates = (1 << n) - 1
                
            mask_LR = 0
            for x in L: mask_LR |= (1 << x)
            for x in R: mask_LR |= (1 << x)
            common_R_candidates &= ~mask_LR
            
            c = common_R_candidates
            while c:
                lb = c & -c
                w = lb.bit_length() - 1
                c ^= lb
                
                covers_new = False
                for u in L:
                    e = (min(w,u), max(w,u))
                    if e in edge_to_idx:
                        idx = edge_to_idx[e]
                        if (rem_mask >> idx) & 1:
                            covers_new = True
                            break
                            
                if covers_new:
                    R.add(w)
                    changed = True

        return L, R

    # Greedy Set Cover
    # We need to find the biclique covering the most remaining edges.
    # Strategy: Sample candidate bicliques.
    # Candidates:
    # 1. For each remaining edge (u, v), expand from ({u}, {v}).
    # 2. For each vertex u, expand from ({u}, N(u) intersected with vertices having remaining edges to u).
    
    while remaining_mask:
        best_L = None
        best_R = None
        best_count = 0
        best_mask = 0
        
        # Get list of remaining edges to iterate
        rem_edges = []
        m = remaining_mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            m ^= lb
            rem_edges.append(idx_to_edge[idx])
            
        # To avoid O(E^2) expansion calls, we can limit candidates or use a better heuristic.
        # Given N=30, E=150, E^2 = 22500 expansions.
        # Each expansion is O(N * (|L|+|R|)).
        # This might be too slow if we do it for every edge.
        
        # Alternative: Pick a random remaining edge? Or the one with highest degree in remaining graph?
        # Let's try picking the edge (u,v) where u has the most remaining neighbors.
        
        # Calculate remaining degrees
        rem_deg = [0] * n
        for idx, (u, v) in enumerate(idx_to_edge):
            if (remaining_mask >> idx) & 1:
                rem_deg[u] += 1
                rem_deg[v] += 1
        
        # Sort remaining edges by sum of degrees (heuristic for larger bicliques)
        # Or just pick the edge with max rem_deg[u] + rem_deg[v]
        
        # Let's sample a few candidates:
        # 1. Edge with max degree sum
        # 2. Top 10 edges by degree sum
        # 3. Star bicliques: For each vertex u, L={u}, R={v | (u,v) in rem}.
        
        candidates = []
        
        # Add star candidates
        for u in range(n):
            if rem_deg[u] == 0: continue
            R = set()
            for v in range(n):
                if u == v: continue
                e = (min(u,v), max(u,v))
                if e in edge_to_idx:
                    idx = edge_to_idx[e]
                    if (remaining_mask >> idx) & 1:
                        R.add(v)
            if R:
                candidates.append(( {u}, R ))
                
        # Add expanded candidates from high-degree edges
        # Sort rem_edges by rem_deg[u] + rem_deg[v]
        scored_edges = [(rem_deg[u] + rem_deg[v], u, v) for (u, v) in rem_edges]
        scored_edges.sort(reverse=True)
        
        # Take top 20 edges
        for score, u, v in scored_edges[:20]:
            L, R = expand_biclique({u}, {v}, remaining_mask)
            if L and R and not (L & R):
                candidates.append((L, R))
        
        # Evaluate candidates
        for L, R in candidates:
            if not L or not R: continue
            if L & R: continue
            # Check validity (should be valid by construction, but just in case)
            # is_valid = True
            # for u in L:
            #   for v in R:
            #     if (min(u,v), max(u,v)) not in edge_to_idx:
            #       is_valid = False
            #       break
            # if not is_valid: continue
            
            # Count coverage
            # We can compute mask and intersect
            # But we can just count directly
            count = 0
            for u in L:
                for v in R:
                    e = (min(u,v), max(u,v))
                    if e in edge_to_idx:
                        idx = edge_to_idx[e]
                        if (remaining_mask >> idx) & 1:
                            count += 1
                            
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R
                # Compute mask for removal
                best_mask = 0
                for u in L:
                    for v in R:
                        e = (min(u,v), max(u,v))
                        if e in edge_to_idx:
                            idx = edge_to_idx[e]
                            best_mask |= (1 << idx)
        
        if best_L is None:
            # Fallback: single edges
            for idx, (u, v) in enumerate(idx_to_edge):
                if (remaining_mask >> idx) & 1:
                    bicliques.append([[int(u)], [int(v)]])
                    remaining_mask &= ~(1 << idx)
            break
        
        bicliques.append([sorted(best_L), sorted(best_R)])
        remaining_mask &= ~best_mask
        
    # Convert to proper types
    result = []
    for L, R in bicliques:
        L_ints = [int(x) for x in L]
        R_ints = [int(x) for x in R]
        result.append([L_ints, R_ints])
    
    return result
# EVOLVE-BLOCK-END
