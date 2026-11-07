"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build edge set for O(1) lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    # Precompute maximal bicliques using a heuristic approach
    # We'll generate candidate bicliques by considering common neighborhoods
    
    candidates = []
    
    # Strategy 1: For each pair of vertices, find common neighbors
    # A biclique (L, R) where L={u,v} and R = N(u) ∩ N(v) \ {u,v}
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edge_set and (v, u) not in edge_set:
                # u and v are not connected, but we can still have a biclique
                # where u is on one side and v on the other? No, they need to be
                # on opposite sides or same side.
                # Actually, for a biclique (L, R), all pairs (l, r) for l in L, r in R
                # must be edges. So if u and v are in L, they don't need to be connected.
                # But if u in L and v in R, they must be connected.
                # Let's think differently.
                pass
    
    # Better strategy: Generate maximal bicliques by expanding from edges
    # Start with each edge (u, v) and try to grow the biclique
    
    # For each edge, try to form a biclique where u is on left and v is on right
    # Then expand by adding vertices that connect to all vertices on the other side
    
    seen_bicliques = set()
    
    def normalize_biclique(L, R):
        # Canonical form: sort both, then put the smaller set first (or lexicographically)
        L = tuple(sorted(L))
        R = tuple(sorted(R))
        if len(L) < len(R) or (len(L) == len(R) and L < R):
            return (L, R)
        else:
            return (R, L)
    
    for u, v in edges:
        # Try to form biclique with u on left, v on right
        # Initial: L = {u}, R = {v}
        # Expand: add w to L if w connects to all in R
        #           add w to R if w connects to all in L
        
        # Use iterative expansion
        L = {u}
        R = {v}
        
        changed = True
        while changed:
            changed = False
            # Try adding to L
            for w in range(n):
                if w not in L and w not in R:
                    # w can be added to L if w connects to all in R
                    if all(w in adj[r] for r in R):
                        L.add(w)
                        changed = True
            # Try adding to R
            for w in range(n):
                if w not in L and w not in R:
                    # w can be added to R if w connects to all in L
                    if all(w in adj[l] for l in L):
                        R.add(w)
                        changed = True
        
        # Also try with u on right, v on left
        L2 = {v}
        R2 = {u}
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w not in L2 and w not in R2:
                    if all(w in adj[r] for r in R2):
                        L2.add(w)
                        changed = True
            for w in range(n):
                if w not in L2 and w not in R2:
                    if all(w in adj[l] for l in L2):
                        R2.add(w)
                        changed = True
        
        for L_c, R_c in [(L, R), (L2, R2)]:
            if len(L_c) >= 1 and len(R_c) >= 1:
                key = normalize_biclique(L_c, R_c)
                if key not in seen_bicliques:
                    seen_bicliques.add(key)
                    candidates.append((list(key[0]), list(key[1])))
    
    # Also add single-vertex bicliques for each edge
    for u, v in edges:
        key = normalize_biclique([u], [v])
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            candidates.append([key[0], key[1]])
    
    # Now apply greedy set cover
    # Each candidate biclique covers a set of edges
    # We need to cover all edges in edge_set
    
    # Map each edge to its index
    edge_list = list(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    total_edges = len(edge_list)
    
    # For each candidate, compute the set of edges it covers
    candidate_edges = []
    for L, R in candidates:
        covered = set()
        for l in L:
            for r in R:
                if l < r:
                    e = (l, r)
                else:
                    e = (r, l)
                if e in edge_to_idx:
                    covered.add(e)
        candidate_edges.append(covered)
    
    # Greedy set cover
    uncovered = set(range(total_edges))
    selected = []
    
    while uncovered:
        best_idx = -1
        best_count = 0
        for i, cov in enumerate(candidate_edges):
            count = len(cov & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            # Should not happen if candidates are complete
            break
        
        selected.append(best_idx)
        uncovered -= candidate_edges[best_idx]
    
    # Build the result
    result = []
    for idx in selected:
        L, R = candidates[idx]
        result.append([list(L), list(R)])
    
    return result

# EVOLVE-BLOCK-END
