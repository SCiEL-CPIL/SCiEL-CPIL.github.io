"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Set of remaining edges as frozensets for easy removal
    remaining_edges = set()
    for u, v in edges:
        remaining_edges.add(frozenset((u, v)))
        
    bicliques = []
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Sort edges by degree sum to pick "central" edges first?
    # Or just iterate.
    # To be deterministic and fast, we can just pick any edge.
    # However, picking edges that are part of large cliques/bicliques is good.
    # Let's just iterate through the list of edges, but we need to check if they are still remaining.
    
    # A better heuristic: At each step, find the edge (u,v) in remaining_edges that maximizes |N(u) cap N(v)|.
    # This is expensive if done naively. 
    # Given n=30, |E|~150, it's very small.
    
    while remaining_edges:
        if time.time() > deadline:
            break
            
        best_edge = None
        best_size = -1
        best_L = set()
        best_R = set()
        
        # Iterate through remaining edges to find the one that yields the largest biclique
        for edge in remaining_edges:
            u, v = tuple(edge)
            # Initial biclique L={u}, R={v}
            # Expand L: nodes connected to all in R
            # Expand R: nodes connected to all in L
            
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                # Expand L
                # Candidates for L: neighbors of all nodes in R
                # Actually, for L to be valid, every node in L must be connected to every node in R.
                # So L subset of intersection of neighbors of all nodes in R.
                # But we also need L to be connected to R? Yes, that's the definition.
                # So new_L_candidates = intersection(adj[x] for x in R)
                # We can add any node from new_L_candidates that is not already in L?
                # Wait, if we add x to L, we must ensure x is connected to all y in R.
                # The intersection of adj[x] for x in R gives all nodes connected to ALL nodes in R.
                # Let's call this CommonNeighborsR.
                # Any node in CommonNeighborsR can be added to L.
                
                # Optimization: We want to maximize size.
                # Let's compute CommonNeighborsR.
                common_R = set.intersection(*[adj[x] for x in R]) if R else set()
                # Nodes in common_R are valid additions to L.
                # But we should only add nodes that are "useful".
                # Actually, adding ANY node from common_R to L is valid.
                # To maximize L, we add all of them?
                # If we add all, L becomes common_R.
                # Then we recompute CommonNeighborsL.
                
                # Let's try to grow L to be as large as possible given R.
                # L_new = common_R
                # But we must keep u in L. u is in common_R because u is connected to all in R (initially v, and we only add valid nodes).
                # Actually, u is connected to v. If we add w to R, we must ensure u is connected to w.
                # So the logic must be symmetric.
                
                # Let's use a standard maximal biclique growth:
                # 1. Compute C_L = intersection of adj[x] for x in L
                # 2. Compute C_R = intersection of adj[x] for x in R
                # 3. We can add any node from C_R to L.
                # 4. We can add any node from C_L to R.
                
                # To get a "maximal" biclique, we should add as many as possible.
                # However, adding a node to L might restrict R, and vice versa.
                # A greedy expansion:
                
                # Let's try to expand L first.
                C_R = set.intersection(*[adj[x] for x in R]) if R else set()
                # Add all nodes in C_R to L?
                # If we add all, L becomes C_R.
                # Then we expand R based on new L.
                
                # This might be too aggressive or too slow if sets are large, but n=30 is tiny.
                
                new_L = C_R
                if len(new_L) > len(L):
                    L = new_L
                    changed = True
                
                C_L = set.intersection(*[adj[x] for x in L]) if L else set()
                new_R = C_L
                if len(new_R) > len(R):
                    R = new_R
                    changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_edge = edge
                best_L = L
                best_R = R
        
        if best_edge is None:
            break
            
        # Add biclique
        bicliques.append([list(best_L), list(best_R)])
        
        # Remove covered edges
        edges_to_remove = set()
        for u in best_L:
            for v in best_R:
                edges_to_remove.add(frozenset((u, v)))
        
        remaining_edges -= edges_to_remove
        
    # If we timed out or failed to cover all, we must cover remaining edges with single-edge bicliques to be valid
    if remaining_edges:
        for edge in remaining_edges:
            u, v = tuple(edge)
            bicliques.append([[u], [v]])
            
    # Ensure types are correct
    result = []
    for L, R in bicliques:
        # Sort for determinism? Not required but good.
        # Ensure they are lists of ints
        result.append([list(L), list(R)])
        
    return result
# EVOLVE-BLOCK-END
