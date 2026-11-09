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
    
    # Precompute neighbors for initial degree calculation
    # But degrees change as we remove edges.
    
    def get_max_biclique(start_u, start_v, rem_edges):
        L = {start_u}
        R = {start_v}
        
        # Expand L and R greedily
        changed = True
        while changed:
            changed = False
            
            # Try to add to L
            candidates_L = []
            for u in range(n):
                if u in L or u in R:
                    continue
                # Check if u can be added to L
                # u must be connected to all vertices in R via remaining edges
                can_add = True
                for v in R:
                    if frozenset((u, v)) not in rem_edges:
                        can_add = False
                        break
                if can_add:
                    # Heuristic: prefer vertices with high degree in rem_edges
                    deg = sum(1 for e in rem_edges if u in e)
                    candidates_L.append((deg, u))
            
            if candidates_L:
                candidates_L.sort(reverse=True)
                # Add the best candidate
                _, best_u = candidates_L[0]
                L.add(best_u)
                changed = True
                
            # Try to add to R
            candidates_R = []
            for v in range(n):
                if v in L or v in R:
                    continue
                # Check if v can be added to R
                # v must be connected to all vertices in L via remaining edges
                can_add = True
                for u in L:
                    if frozenset((u, v)) not in rem_edges:
                        can_add = False
                        break
                if can_add:
                    deg = sum(1 for e in rem_edges if v in e)
                    candidates_R.append((deg, v))
            
            if candidates_R:
                candidates_R.sort(reverse=True)
                _, best_v = candidates_R[0]
                R.add(best_v)
                changed = True
                
        return L, R

    bicliques = []
    
    # Iterate until no edges remain
    while remaining_edges:
        # Pick an edge to start a new biclique
        # Strategy: Pick an edge where the endpoints have high degree in the remaining graph
        # or just pick the first one. Let's pick the one with max (deg(u) + deg(v)).
        
        # Calculate degrees in remaining graph
        degrees = {i: 0 for i in range(n)}
        for e in remaining_edges:
            u, v = tuple(e)
            degrees[u] += 1
            degrees[v] += 1
            
        best_edge = None
        best_score = -1
        for e in remaining_edges:
            u, v = tuple(e)
            score = degrees[u] + degrees[v]
            if score > best_score:
                best_score = score
                best_edge = e
                
        u, v = tuple(best_edge)
        
        # Find the largest biclique containing this edge
        L, R = get_max_biclique(u, v, remaining_edges)
        
        # Ensure L and R are disjoint (they should be, as we check u in R etc)
        # Remove all edges in L x R from remaining_edges
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
