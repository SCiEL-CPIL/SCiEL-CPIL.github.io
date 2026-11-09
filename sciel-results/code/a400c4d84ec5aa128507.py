"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Track uncovered edges
    uncovered = set()
    for u, v in edges:
        uncovered.add((min(u,v), max(u,v)))
    
    if not uncovered:
        return []
    
    bicliques = []
    
    while uncovered:
        if time.time() > deadline:
            # Fallback: cover remaining edges one by one
            for e in list(uncovered):
                u, v = e
                bicliques.append([[u], [v]])
            break
        
        # Pick an uncovered edge - use the one with highest combined degree
        best_edge = None
        best_score = -1
        for e in uncovered:
            u, v = e
            score = len(adj[u]) + len(adj[v])
            if score > best_score:
                best_score = score
                best_edge = e
        
        u, v = best_edge
        
        # Find maximal biclique containing edge (u,v) in uncovered subgraph
        # u goes on left, v goes on right initially
        L = {u}
        R = {v}
        
        # Iteratively expand
        changed = True
        while changed:
            changed = False
            # Expand L: add vertices connected to ALL in R
            new_L = set()
            for w in range(n):
                if w in L:
                    new_L.add(w)
                    continue
                # Check if w is connected to all in R via uncovered edges
                all_connected = True
                for r in R:
                    a, b = min(w, r), max(w, r)
                    if (a, b) not in uncovered:
                        all_connected = False
                        break
                if all_connected:
                    new_L.add(w)
                    changed = True
            L = new_L
            
            # Expand R: add vertices connected to ALL in L
            new_R = set()
            for w in range(n):
                if w in R:
                    new_R.add(w)
                    continue
                all_connected = True
                for l in L:
                    a, b = min(w, l), max(w, l)
                    if (a, b) not in uncovered:
                        all_connected = False
                        break
                if all_connected:
                    new_R.add(w)
                    changed = True
            R = new_R
        
        # Ensure L and R are disjoint (they should be since no self-loops)
        R = R - L
        
        if not L or not R:
            # Fallback: just use the single edge
            bicliques.append([[u], [v]])
            uncovered.discard((min(u,v), max(u,v)))
            continue
        
        # Add this biclique
        bicliques.append([list(L), list(R)])
        
        # Remove all edges in this biclique from uncovered
        for l in L:
            for r in R:
                a, b = min(l, r), max(l, r)
                uncovered.discard((a, b))
    
    # Convert to proper types
    result = []
    for bic in bicliques:
        left = list(bic[0])
        right = list(bic[1])
        # Ensure nonempty and disjoint
        if left and right and not (set(left) & set(right)):
            result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
