"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 50.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Track uncovered edges as a set of frozensets for easy lookup
    uncovered = set()
    for u, v in edges:
        uncovered.add(frozenset([u, v]))
    
    bicliques = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Compute uncovered degree for each vertex
        uncovered_deg = [0] * n
        for e in uncovered:
            a, b = tuple(e)
            uncovered_deg[a] += 1
            uncovered_deg[b] += 1
        
        # Find vertex with max uncovered degree
        best_u = max(range(n), key=lambda x: uncovered_deg[x])
        
        if uncovered_deg[best_u] == 0:
            break
        
        # Get uncovered neighbors of best_u
        unc_neighbors = set()
        for e in uncovered:
            a, b = tuple(e)
            if a == best_u:
                unc_neighbors.add(b)
            elif b == best_u:
                unc_neighbors.add(a)
        
        if not unc_neighbors:
            break
        
        # Try to find a larger biclique using multi-anchor
        # For each pair of neighbors, check if they have common uncovered neighbors
        # that include best_u's other neighbors
        
        # Start with star: [best_u, unc_neighbors]
        # Try to expand: find subsets of unc_neighbors that all connect to another vertex
        
        # Simple heuristic: try to find a vertex w (not best_u) that connects to many of unc_neighbors
        # and also connects to best_u (it does, since w in unc_neighbors means edge best_u-w exists)
        # Actually w is in unc_neighbors, so edge (best_u, w) exists.
        # For w to be in the left side with best_u, we need all of w's neighbors in right to also be in right.
        
        # Let's try: for each w in unc_neighbors, compute common_uncovered_neighbors(best_u, w)
        # = neighbors of w that are also uncovered neighbors of best_u
        # Then biclique [best_u, w] x common could cover more edges
        
        best_biclique = None
        best_count = len(unc_neighbors)  # star covers this many edges
        
        # Try each w in unc_neighbors as a second anchor
        for w in unc_neighbors:
            # Common uncovered neighbors of best_u and w
            common = unc_neighbors & adj[w]
            # The biclique [best_u, w] x common covers:
            # For left={best_u, w}, right=common
            # Edges: (best_u, c) for c in common, (w, c) for c in common
            # But we need all these to be in uncovered
            # Check: (best_u, c) is in uncovered for all c in common? Yes, common subset of unc_neighbors
            # Check: (w, c) is in uncovered for all c in common? Need to verify
            valid_common = set()
            for c in common:
                if frozenset([w, c]) in uncovered:
                    valid_common.add(c)
            
            if len(valid_common) > best_count:
                best_count = len(valid_common)
                best_biclique = ([best_u, w], valid_common)
        
        # Also try pairs of vertices from unc_neighbors
        unc_neigh_list = list(unc_neighbors)
        for i in range(len(unc_neigh_list)):
            for j in range(i+1, len(unc_neigh_list)):
                w1 = unc_neigh_list[i]
                w2 = unc_neigh_list[j]
                # Common uncovered neighbors
                common = unc_neighbors & adj[w1] & adj[w2]
                valid_common = set()
                for c in common:
                    if frozenset([w1, c]) in uncovered and frozenset([w2, c]) in uncovered:
                        valid_common.add(c)
                if len(valid_common) > best_count:
                    best_count = len(valid_common)
                    best_biclique = ([w1, w2], valid_common)
        
        if best_biclique:
            left, right = best_biclique
            right = list(right)
        else:
            left = [best_u]
            right = list(unc_neighbors)
        
        # Remove all cross edges from uncovered
        for l in left:
            for r in right:
                uncovered.discard(frozenset([l, r]))
        
        bicliques.append([left, right])
    
    # Handle remaining edges if timed out
    if uncovered:
        remaining = list(uncovered)
        assigned = set()
        for e in remaining:
            a, b = tuple(e)
            for v in [a, b]:
                if v not in assigned:
                    neighbors = set()
                    for e2 in remaining:
                        x, y = tuple(e2)
                        if x == v:
                            neighbors.add(y)
                        elif y == v:
                            neighbors.add(x)
                    if neighbors:
                        bicliques.append([[v], list(neighbors)])
                        assigned.add(v)
                    break
    
    # Ensure correct types
    result = []
    for bic in bicliques:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
