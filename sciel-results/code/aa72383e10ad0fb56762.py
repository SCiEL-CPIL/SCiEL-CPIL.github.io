import time
import random

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Use a set for fast lookup and removal
    remaining_edges = set(tuple(sorted(e)) for e in edges)
    # We also need to know the original orientation? No, bicliques are unordered pairs of sets.
    # But the output requires [left, right]. The order within left/right doesn't matter, 
    # but the pair (left, right) defines the biclique. 
    # The task says "partition all pinned graph edges". 
    # An edge (u,v) is in the biclique if u in L and v in R OR v in L and u in R.
    # So we can treat edges as undirected for the algorithm.
    
    bicliques = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit, leaving 10s margin
    
    while remaining_edges:
        if time.time() > deadline:
            # Fallback: add remaining edges as individual bicliques (stars or single edges)
            # To be safe and valid, we can just create a biclique for each remaining edge
            # as [u, [v]] or [[u, v], []] ? No, sides must be non-empty.
            # So [u, [v]] is a valid biclique for edge (u,v).
            for e in remaining_edges:
                u, v = e
                bicliques.append([[u], [v]])
            break
            
        # Find the best biclique in the remaining graph
        best_biclique = None
        best_edge_count = 0
        
        # Heuristic: Try expanding from a few promising seeds
        # Sort edges by degree sum to find dense areas
        # Calculate degrees in remaining graph
        deg = {i: 0 for i in range(n)}
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
            
        # Pick top K edges as seeds
        # K=10 is small, but helps find good bicliques
        sorted_edges = sorted(remaining_edges, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        seeds = sorted_edges[:10]
        
        for u, v in seeds:
            # Expand (u,v) to a maximal biclique
            L = {u}
            R = {v}
            
            # Iteratively add vertices
            changed = True
            while changed:
                changed = False
                # Find candidates for L
                # x must be connected to all in R
                # x not in L or R
                L_candidates = []
                for x in range(n):
                    if x in L or x in R:
                        continue
                    # Check if x is connected to all in R
                    is_connected = True
                    for y in R:
                        e1 = (min(x, y), max(x, y))
                        if e1 not in remaining_edges:
                            is_connected = False
                            break
                    if is_connected:
                        L_candidates.append(x)
                        
                if L_candidates:
                    # Pick the one with highest degree
                    best_x = max(L_candidates, key=lambda x: deg[x])
                    L.add(best_x)
                    changed = True
                    
                # Find candidates for R
                R_candidates = []
                for y in range(n):
                    if y in L or y in R:
                        continue
                    is_connected = True
                    for x in L:
                        e1 = (min(x, y), max(x, y))
                        if e1 not in remaining_edges:
                            is_connected = False
                            break
                    if is_connected:
                        R_candidates.append(y)
                        
                if R_candidates:
                    best_y = max(R_candidates, key=lambda y: deg[y])
                    R.add(best_y)
                    changed = True
            
            # Calculate number of edges in this biclique
            # All pairs (l, r) with l in L, r in R must be in remaining_edges
            # By construction they are, but let's verify and count
            edge_count = 0
            valid = True
            for l in L:
                for r in R:
                    e1 = (min(l, r), max(l, r))
                    if e1 in remaining_edges:
                        edge_count += 1
                    else:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and edge_count > best_edge_count:
                best_edge_count = edge_count
                best_biclique = (list(L), list(R))
                
        if best_biclique is None:
            # Should not happen if remaining_edges is not empty
            # Fallback to single edge
            u, v = next(iter(remaining_edges))
            best_biclique = ([u], [v])
            best_edge_count = 1
            
        # Add to result
        bicliques.append(best_biclique)
        
        # Remove edges from remaining_edges
        L, R = best_biclique
        for l in L:
            for r in R:
                e1 = (min(l, r), max(l, r))
                remaining_edges.discard(e1)
                
    # Convert to required format: list of [left, right]
    # Ensure types are list
    final_result = []
    for l, r in bicliques:
        final_result.append([list(l), list(r)])
        
    return final_result
# EVOLVE-BLOCK-END
