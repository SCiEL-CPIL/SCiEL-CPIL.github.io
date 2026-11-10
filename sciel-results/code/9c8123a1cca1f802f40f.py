import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and bitmask
    adj = [set() for _ in range(n)]
    adj_mask = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Function to check if a set of vertices forms a clique using bitmasks
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if not (adj_mask[u] >> v) & 1:
                    return False
        return True
    
    # Bron-Kerbosch algorithm to find maximum clique
    max_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal max_clique
        if time.time() > deadline:
            return
        
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Choose pivot to minimize branching
        pivot = None
        max_deg = -1
        for v in P_set:
            deg = len(P_set & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            if not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
            return
            
        # Branch on vertices in P that are NOT neighbors of pivot
        candidates = list(P_set - adj[pivot])
        
        for v in candidates:
            if time.time() > deadline:
                return
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_set.remove(v)
            X.add(v)
            
    # Initialize
    R = set()
    P_set = set(range(n))
    X = set()
    bron_kerbosch(R, P_set, X)
    
    # Now we have the maximum clique. Let's try to find a larger set of branches
    # where we can route vertex-disjoint paths between all pairs.
    # For a clique, this is trivial (each pair has a direct edge).
    # But maybe we can do better by not requiring a clique?
    # Actually, for the paths to be vertex-disjoint (except at endpoints),
    # we need Menger's theorem conditions.
    # Let's stick with the clique approach first since it's guaranteed to work.
    
    if len(max_clique) >= 2:
        branches = list(max_clique)
        branches.sort()
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                u, v = branches[i], branches[j]
                paths.append([u, v])
        
        # Validate and return
        return {'branches': [int(b) for b in branches], 'paths': [list(p) for p in paths]}
    
    # Fallback: Find any edge
    if edges:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
    # Fallback: Two isolated vertices
    return {'branches': [0, 1], 'paths': [[0, 1]]}
# EVOLVE-BLOCK-END
