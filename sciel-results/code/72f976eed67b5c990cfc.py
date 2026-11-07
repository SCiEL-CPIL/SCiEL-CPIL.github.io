"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    # Bron-Kerbosch with pivoting to find maximum clique
    max_clique = []
    
    def bronk(R, P_set, X_set):
        nonlocal max_clique
        if time.time() > deadline:
            return
        
        if not P_set and not X_set:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Choose pivot u from P_set U X_set to minimize branching
        # We want u with max degree in P_set
        U = P_set.union(X_set)
        if not U:
            return
            
        # Find pivot with max intersection with P_set
        pivot = None
        max_intersect = -1
        for u in U:
            inter = len(P_set.intersection(adj[u]))
            if inter > max_intersect:
                max_intersect = inter
                pivot = u
                
        # Branch on vertices in P_set \ N(pivot)
        candidates = list(P_set.difference(adj[pivot]))
        
        for v in candidates:
            # Recursive call
            bronk(R.union({v}), P_set.intersection(adj[v]), X_set.intersection(adj[v]))
            
            # Move v from P to X
            P_set.remove(v)
            X_set.add(v)
            
            if time.time() > deadline:
                return

    # Initial call
    all_vertices = set(range(n))
    bronk(set(), all_vertices, set())
    
    # If no clique found or clique size < 2, fallback to an edge
    if len(max_clique) < 2:
        if edges:
            u, v = edges[0]
            max_clique = [u, v]
        else:
            # Should not happen based on problem statement
            max_clique = [0, 1]
            
    branches = [int(v) for v in max_clique]
    
    # Construct paths for each pair of branches
    # Since branches form a clique, direct edges exist
    paths = []
    for u, v in combinations(branches, 2):
        # Ensure u, v are adjacent (they should be if it's a clique)
        if v in adj[u]:
            paths.append([u, v])
        else:
            # This should not happen if max_clique is a valid clique
            # But just in case, we need to find a path or return invalid
            # For safety, if this happens, we might have a bug in Bron-Kerbosch
            # Or the graph is not fully connected in a way that supports this.
            # However, Bron-Kerbosch only finds cliques, so edges must exist.
            pass
            
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
