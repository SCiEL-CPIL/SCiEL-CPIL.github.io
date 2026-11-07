"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _count_distinct_degrees(vertices, adj):
    vset = set(vertices)
    degrees = []
    for v in vertices:
        deg = sum(1 for u in adj[v] if u in vset)
        degrees.append(deg)
    return len(set(degrees))

def construct():
    deadline = time.time() + 80
    n = P['n']
    adj = _build_adj()
    
    # Precompute degree of each vertex in the full graph
    full_degrees = [len(adj[i]) for i in range(n)]
    
    best_count = 0
    best_vertices = list(range(8))
    
    # Try all subsets of size 8 to 12
    for size in range(8, min(13, n + 1)):
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            count = _count_distinct_degrees(combo, adj)
            if count > best_count:
                best_count = count
                best_vertices = list(combo)
        if time.time() > deadline:
            break
    
    # If we haven't found 8 distinct degrees, try larger subsets or different strategies
    if best_count < 8:
        # Try all subsets of size 13 to 17
        for size in range(13, n + 1):
            for combo in combinations(range(n), size):
                if time.time() > deadline:
                    break
                count = _count_distinct_degrees(combo, adj)
                if count > best_count:
                    best_count = count
                    best_vertices = list(combo)
            if time.time() > deadline:
                break
    
    return list(best_vertices)
# EVOLVE-BLOCK-END
