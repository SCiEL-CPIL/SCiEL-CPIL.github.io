"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build global adjacency sets for fast lookup
    global_adj = [set() for _ in range(n)]
    for u, v in edges:
        global_adj[u].add(v)
        global_adj[v].add(u)
        
    # Remaining edges as a set of tuples (min, max)
    remaining_edges = set((min(u,v), max(u,v)) for u,v in edges)
    
    # Helper to check if edge (u,v) is in remaining
    def is_rem_edge(u, v):
        return (min(u,v), max(u,v)) in remaining_edges

    def get_maximal_biclique(L_seed, R_seed, rem_edges):
        """
        Expands L_seed and R_seed to form a maximal biclique 
        using only edges in rem_edges.
        L and R must be disjoint.
        """
        L = set(L_seed)
        R = set(R_seed)
        
        # Ensure disjointness
        if L & R:
            return None, None
            
        changed = True
        while changed:
            changed = False
            # Expand L
            new_L = set(L)
            for w in range(n):
                if w in L or w in R:
                    continue
                # w can join L if w is connected to all in R via rem_edges
                can_add = True
                for v in R:
                    if not is_rem_edge(w, v):
                        can_add = False
                        break
                if can_add:
                    new_L.add(w)
            if new_L != L:
                L = new_L
                changed = True
                
            # Expand R
            new_R = set(R)
            for w in range(n):
                if w in L or w in R:
                    continue
                # w can join R if w is connected to all in L via rem_edges
                can_add = True
                for u in L:
                    if not is_rem_edge(w, u):
                        can_add = False
                        break
                if can_add:
                    new_R.add(w)
            if new_R != R:
                R = new_R
                changed = True
                
        return L, R

    bicliques = []
    
    # Greedy loop
    while remaining_edges:
        # Calculate degrees in remaining graph
        deg = [0] * n
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
            
        # Find top K vertices by degree
        sorted_vertices = sorted(range(n), key=lambda x: deg[x], reverse=True)
        
        best_biclique = None
        best_count = 0
        
        # Try seeds from top 10 vertices
        for u in sorted_vertices[:10]:
            if deg[u] == 0:
                continue
            for v in global_adj[u]:
                if is_rem_edge(u, v):
                    # Try to build maximal biclique with L containing u, R containing v
                    L, R = get_maximal_biclique({u}, {v}, remaining_edges)
                    if L is None or R is None:
                        continue
                    
                    # Calculate how many remaining edges this biclique covers
                    covered_edges = set()
                    for l in L:
                        for r in R:
                            covered_edges.add((min(l,r), max(l,r)))
                            
                    # Intersect with remaining to be safe
                    valid_covered = covered_edges & remaining_edges
                    count = len(valid_covered)
                    
                    if count > best_count:
                        best_count = count
                        best_biclique = (L, R, valid_covered)
                        
        if best_biclique is None:
            # Fallback: take any remaining edge
            u, v = next(iter(remaining_edges))
            bicliques.append([[int(u)], [int(v)]])
            remaining_edges.discard((u, v))
        else:
            L, R, covered = best_biclique
            bicliques.append([sorted(L), sorted(R)])
            remaining_edges -= covered
            
    # Convert to proper types
    result = []
    for L, R in bicliques:
        L_ints = [int(x) for x in L]
        R_ints = [int(x) for x in R]
        result.append([L_ints, R_ints])
        
    return result
# EVOLVE-BLOCK-END
