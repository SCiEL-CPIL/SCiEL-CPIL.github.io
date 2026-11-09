"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Normalize edges to frozensets for easy lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add(frozenset((u, v)))
    
    remaining_edges = set(edge_set)
    bicliques = []
    
    while remaining_edges:
        # Build adjacency list for remaining edges
        adj = {i: set() for i in range(n)}
        for e in remaining_edges:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
            
        # Select edge with maximum common neighbors
        best_edge = None
        max_common = -1
        for e in remaining_edges:
            u, v = tuple(e)
            # Common neighbors are vertices connected to both u and v in the remaining graph
            # Note: u and v are connected, so they are in each other's adj.
            # We want vertices w such that (u,w) in rem and (v,w) in rem.
            common = len(adj[u] & adj[v])
            if common > max_common:
                max_common = common
                best_edge = e
                
        if best_edge is None:
            break
            
        u, v = tuple(best_edge)
        
        # Expand to maximal biclique containing (u,v)
        L = {u}
        R = {v}
        
        changed = True
        while changed:
            changed = False
            
            # Find candidates for L
            candidates_L = []
            for x in range(n):
                if x in L or x in R:
                    continue
                # x must be connected to all in R
                if all(frozenset((x, y)) in remaining_edges for y in R):
                    # Heuristic: prefer x that is connected to many potential R candidates
                    # or just add it
                    candidates_L.append(x)
            
            # Find candidates for R
            candidates_R = []
            for y in range(n):
                if y in L or y in R:
                    continue
                # y must be connected to all in L
                if all(frozenset((x, y)) in remaining_edges for x in L):
                    candidates_R.append(y)
                    
            # Add all valid candidates? Or just one?
            # Adding all might be safer for "maximal" if they don't conflict.
            # But adding one might allow more future additions.
            # Let's add all candidates that are "safe".
            # Actually, if x is in candidates_L, it is connected to all current R.
            # If we add x to L, R might shrink? No, R is fixed in this step.
            # But adding x to L makes it harder for new vertices to join R.
            # Greedy: Add the vertex that allows the most future growth?
            # Simpler: Add all candidates to L and R simultaneously?
            # If we add x to L and y to R, we need (x,y) to be an edge.
            # Let's stick to adding one at a time, prioritizing high degree in remaining graph.
            
            if candidates_L:
                # Pick candidate with max degree in remaining graph
                best_x = max(candidates_L, key=lambda x: len(adj[x]))
                L.add(best_x)
                changed = True
                # Update adj? No, adj is static for this iteration of outer loop.
                # But L and R change. The condition "connected to all in R" uses remaining_edges.
                # So we don't need to update adj for the *check*, but the *set* of valid candidates changes.
                
            if candidates_R:
                best_y = max(candidates_R, key=lambda y: len(adj[y]))
                R.add(best_y)
                changed = True
        
        # Remove edges covered by L x R
        for lu in L:
            for rv in R:
                rem_edge = frozenset((lu, rv))
                if rem_edge in remaining_edges:
                    remaining_edges.remove(rem_edge)
                    
        bicliques.append([L, R])
        
    # Convert to lists of ints
    result = []
    for L, R in bicliques:
        L_list = [int(x) for x in sorted(L)]
        R_list = [int(x) for x in sorted(R)]
        result.append([L_list, R_list])
        
    return result
# EVOLVE-BLOCK-END
