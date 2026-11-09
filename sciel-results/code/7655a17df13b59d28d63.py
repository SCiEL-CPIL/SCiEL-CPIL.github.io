import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Degeneracy ordering
    # 1. Compute degrees
    degrees = [len(adj[i]) for i in range(n)]
    # 2. Iteratively remove vertex with min degree
    order = []
    remaining = set(range(n))
    while remaining:
        # Find vertex with min degree in remaining
        min_v = min(remaining, key=lambda v: len(adj[v] & remaining))
        order.append(min_v)
        remaining.remove(min_v)
        
    # Reverse order for Bron-Kerbosch (usually works better to start from high degree end or use specific ordering)
    # Actually, standard Bron-Kerbosch with pivot works well on the original graph too, 
    # but degeneracy ordering helps bound the search space.
    # We will process vertices in the order of increasing degree (degeneracy order)
    
    max_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal max_clique
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Pivot selection: choose vertex u in P_set U X with max |N(u) & P_set|
        U = P_set | X
        if U:
            # Pick pivot with max neighbors in P_set
            pivot = max(U, key=lambda u: len(adj[u] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
            
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
            
    # Initialize
    bron_kerbosch(set(), set(range(n)), set())
    
    # If no clique found (shouldn't happen), fallback to a single edge
    if not max_clique:
         u, v = edges[0]
         max_clique = [u, v]
         
    # Ensure at least 2 branches
    if len(max_clique) < 2:
         u, v = edges[0]
         max_clique = [u, v]
         
    branches = list(max_clique)
    # Generate paths for all pairs
    paths = []
    for u, v in combinations(branches, 2):
        # Since they form a clique, u and v are adjacent
        # Path is just [u, v]
        paths.append([u, v])
        
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
