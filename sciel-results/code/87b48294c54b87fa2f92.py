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

def _distinct_degrees(subset, adj):
    """Count distinct degrees in induced subgraph on subset."""
    s = set(subset)
    degs = []
    for v in subset:
        d = len(adj[v] & s)
        degs.append(d)
    return len(set(degs))

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s (and 90s kill)
    
    n = P['n']
    adj = _build_adj()
    min_k = P['minimum_vertices']
    
    best_count = 0
    best_subset = None
    
    # Try all subsets of size >= min_k
    # 2^17 = 131072, very manageable
    # But we only need subsets of size >= 8
    # Let's iterate by size from 17 down to 8
    for k in range(n, min_k - 1, -1):
        if time.time() > deadline:
            break
        for combo in combinations(range(n), k):
            if time.time() > deadline:
                break
            cnt = _distinct_degrees(combo, adj)
            if cnt > best_count:
                best_count = cnt
                best_subset = combo
                # Can't do better than k distinct degrees
                if best_count >= k:
                    break
        if best_count >= k:
            break
    
    if best_subset is None:
        return list(range(min_k))
    
    return list(best_subset)
# EVOLVE-BLOCK-END
