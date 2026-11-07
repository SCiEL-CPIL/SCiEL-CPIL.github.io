import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Normalize edges to tuples for set operations
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    def get_biclique_edges(L, R):
        # Returns set of edges in the biclique
        # Edges are (u, v) with u in L, v in R
        # We need to ensure they are in the original graph? 
        # The expansion logic ensures this.
        res = set()
        for u in L:
            for v in R:
                res.add((u, v) if u < v else (v, u))
        return res

    def expand_biclique(seed_u, seed_v, remaining_edges):
        L = {seed_u}
        R = {seed_v}
        changed = True
        while changed:
            changed = False
            # Expand L
            new_L = set(L)
            for w in range(n):
                if w not in L and w not in R:
                    # Check if w is connected to all in R in the ORIGINAL graph
                    # AND the edges (w, r) must be in remaining_edges?
                    # No, the biclique must cover edges that are in remaining_edges.
                    # If we add w to L, we cover edges (w, r) for all r in R.
                    # These edges must be in remaining_edges.
                    if all((w, r) in remaining_edges or (r, w) in remaining_edges for r in R):
                         # Also must be a host edge
                         if all(r in adj[w] for r in R):
                             new_L.add(w)
                             changed = True
            L = new_L
            
            # Expand R
            new_R = set(R)
            for w in range(n):
                if w not in L and w not in R:
                    if all((w, l) in remaining_edges or (l, w) in remaining_edges for l in L):
                         if all(l in adj[w] for l in L):
                             new_R.add(w)
                             changed = True
            R = new_R
        return L, R

    def solve_greedy(seed_order):
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            best_biclique = None
            best_size = -1
            best_edges = set()
            
            # Try seeds from the provided order
            for u, v in seed_order:
                # Only consider if edge is still remaining
                if (u, v) not in remaining and (v, u) not in remaining:
                    continue
                    
                L, R = expand_biclique(u, v, remaining)
                
                # Generate edges covered
                b_edges = set()
                for lu in L:
                    for rv in R:
                        e = (lu, rv) if lu < rv else (rv, lu)
                        b_edges.add(e)
                        
                # Check validity: all b_edges must be in remaining
                if b_edges.issubset(remaining):
                    size = len(b_edges)
                    if size > best_size:
                        best_size = size
                        best_biclique = [list(L), list(R)]
                        best_edges = b_edges
            
            if best_biclique is None:
                # Fallback
                u, v = next(iter(remaining))
                best_biclique = [[u], [v]]
                best_edges = {(u, v) if u < v else (v, u)}
            
            remaining -= best_edges
            bicliques.append(best_biclique)
            
        return bicliques

    start_time = time.time()
    deadline = start_time + 50 # 50s limit
    
    best_solution = None
    best_count = float('inf')
    
    # Initial deterministic run
    initial_order = list(edge_set)
    sol = solve_greedy(initial_order)
    best_solution = sol
    best_count = len(sol)
    
    # Random restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Shuffle edge order
        shuffled = list(edge_set)
        random.shuffle(shuffled)
        
        sol = solve_greedy(shuffled)
        if len(sol) < best_count:
            best_count = len(sol)
            best_solution = sol
            
        # Optional: if we found a very small number, we can stop or continue
        if best_count <= 5: # Heuristic threshold
            break

    return best_solution
