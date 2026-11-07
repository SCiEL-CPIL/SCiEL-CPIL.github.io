"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    def time_up():
        return time.time() >= deadline
    
    # Find maximum clique using Bron-Kerbosch with pivot
    def find_max_clique():
        best_clique = []
        
        def bronk(R, P_set, X):
            nonlocal best_clique
            if time_up():
                return
            if not P_set and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            # Pivot selection: choose vertex in P_set ∪ X with max neighbors in P_set
            P_X = P_set | X
            if not P_X:
                return
            pivot = max(P_X, key=lambda v: len(adj[v] & P_set))
            # For each vertex in P_set not adjacent to pivot
            candidates = [v for v in P_set if v not in adj[pivot]]
            for v in candidates:
                if time_up():
                    return
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set = P_set - {v}
                X = X | {v}
        
        bronk(set(), set(range(n)), set())
        return best_clique
    
    # Try to find a large clique
    max_clique = find_max_clique()
    
    if len(max_clique) >= 2:
        # A clique gives us a valid solution: all pairs are adjacent
        branches = list(max_clique)
        paths = []
        for u, v in combinations(branches, 2):
            paths.append([u, v])
        return {'branches': branches, 'paths': paths}
    
    # Fallback: find any edge
    u, v = edges[0]
    return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
