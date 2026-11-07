"""Biclique partition via greedy maximal biclique set-cover."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _find_max_biclique(remaining_edges, adj, n):
    """Find a large biclique from remaining edges using greedy expansion."""
    if not remaining_edges:
        return None, None
    
    # Score each edge by potential: product of degrees in remaining graph
    best_edge = None
    best_score = -1
    for (u, v) in remaining_edges:
        deg_u = len(adj[u])
        deg_v = len(adj[v])
        score = deg_u * deg_v
        if score > best_score:
            best_score = score
            best_edge = (u, v)
    
    if best_edge is None:
        return None, None
    
    u0, v0 = best_edge
    L = {u0}
    R = {v0}
    
    # Greedy expansion: repeatedly add vertex to L or R that maximizes coverage
    changed = True
    while changed:
        changed = False
        # Try adding to L: a vertex w not in L∪R such that w is connected to all of R
        best_l_gain = 0
        best_l_v = None
        for w in range(n):
            if w in L or w in R:
                continue
            # w must be connected to all vertices in R (in remaining edges)
            ok = True
            for r in R:
                if (min(w,r), max(w,r)) not in remaining_edges:
                    ok = False
                    break
            if not ok:
                continue
            # Gain: number of remaining edges from w to R
            gain = len(R)
            if gain > best_l_gain:
                best_l_gain = gain
                best_l_v = w
        if best_l_v is not None:
            L.add(best_l_v)
            changed = True
            continue
        
        # Try adding to R: a vertex w not in L∪R such that w is connected to all of L
        best_r_gain = 0
        best_r_v = None
        for w in range(n):
            if w in L or w in R:
                continue
            ok = True
            for l in L:
                if (min(l,w), max(l,w)) not in remaining_edges:
                    ok = False
                    break
            if not ok:
                continue
            gain = len(L)
            if gain > best_r_gain:
                best_r_gain = gain
                best_r_v = w
        if best_r_v is not None:
            R.add(best_r_v)
            changed = True
    
    if not L or not R:
        return None, None
    return L, R


def construct():
    n = P['n']
    # Build adjacency as set of frozenset pairs for O(1) lookup
    edge_set = set()
    adj = [set() for _ in range(n)]
    for (a, b) in P['edges']:
        edge_set.add((min(a,b), max(a,b)))
        adj[a].add(b)
        adj[b].add(a)
    
    remaining = set(edge_set)
    bicliques = []
    
    max_iter = 200
    it = 0
    while remaining and it < max_iter:
        it += 1
        L, R = _find_max_biclique(remaining, adj, n)
        if L is None:
            # Fallback: single edge
            e = next(iter(remaining))
            u, v = e
            bicliques.append([[u], [v]])
            remaining.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            continue
        
        L_list = sorted(L)
        R_list = sorted(R)
        bicliques.append([L_list, R_list])
        
        # Remove all cross edges from remaining and adj
        for u in L_list:
            for v in R_list:
                e = (min(u,v), max(u,v))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
    
    # Handle any leftover edges (shouldn't happen but safety)
    for e in list(remaining):
        u, v = e
        bicliques.append([[u], [v]])
    
    # Convert to pure Python types
    result = []
    for (L, R) in bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
    
    return result
# EVOLVE-BLOCK-END
