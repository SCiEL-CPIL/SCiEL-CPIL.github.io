"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    max_clique = []
    
    def bron_kerbosch(r, p, x):
        nonlocal max_clique
        if not p and not x:
            if len(r) > len(max_clique):
                max_clique = list(r)
            return
        
        # Pivot selection: choose vertex in P union X with max degree in P
        pivot = None
        max_deg = -1
        for v in p:
            deg = len(adj[v] & p)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        if pivot is None:
            for v in x:
                deg = len(adj[v] & p)
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
        
        if pivot is not None:
            candidates = [v for v in p if pivot not in adj[v]]
        else:
            candidates = list(p)
            
        for v in candidates:
            bron_kerbosch(r | {v}, p & adj[v], x & adj[v])
            p.remove(v)
            x.add(v)
            
    bron_kerbosch(set(), set(range(n)), set())
    
    # Fallback if no clique found (should not happen for n>=1)
    if not max_clique:
        max_clique = [0, 1] if (0, 1) in [(u, v) for u, v in P['edges']] else [0]
        if len(max_clique) < 2:
             # Find any edge
             for u, v in P['edges']:
                 max_clique = [u, v]
                 break

    branches = max_clique
    paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            u = branches[i]
            v = branches[j]
            # Since it's a clique, u and v are adjacent
            paths.append([u, v])
            
    return {
        'branches': branches,
        'paths': paths
    }
# EVOLVE-BLOCK-END
