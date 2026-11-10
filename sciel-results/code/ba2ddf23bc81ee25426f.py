import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v) if u < v else (v, u))
        
    # Function to check if a set of vertices forms a clique
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    return False
        return True

    # Bron-Kerbosch algorithm to find maximum clique
    # We want the largest clique
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Choose pivot to minimize branching
        # Heuristic: choose vertex with max degree in P
        pivot = None
        max_deg = -1
        for v in P:
            deg = len(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            # P is empty
            if not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
            return
            
        # Branch on vertices in P that are NOT neighbors of pivot
        # This reduces the number of recursive calls
        candidates = list(P - adj[pivot])
        
        for v in candidates:
            if time.time() > deadline:
                return
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
            
    # Initialize
    R = set()
    P = set(range(n))
    X = set()
    bron_kerbosch(R, P, X)
    
    if len(max_clique) >= 2:
        branches = list(max_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                u, v = branches[i], branches[j]
                # Since it's a clique, the path is just the edge
                paths.append([u, v])
        return {'branches': branches, 'paths': paths}
    
    # Fallback: Find any edge
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    # Fallback: Two isolated vertices (should not happen for this graph)
    return {'branches': [0, 1], 'paths': [[0, 1]]}
# EVOLVE-BLOCK-END
