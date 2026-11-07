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

def _distinct_degrees(vertices, adj):
    """Count distinct degrees in the induced subgraph on vertices."""
    vset = set(vertices)
    degrees = []
    for v in vertices:
        deg = sum(1 for u in vset if u != v and u in adj[v])
        degrees.append(deg)
    return len(set(degrees))

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = P['n']
    min_v = P['minimum_vertices']
    adj = _build_adj()
    
    # Precompute adjacency as bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    # Brute-force all subsets of size 8 to 17
    # Total combinations: C(17,8)+C(17,9)+...+C(17,17)
    # C(17,8) = 24310, C(17,9) = 24310, C(17,10) = 19448, C(17,11) = 12376,
    # C(17,12) = 6188, C(17,13) = 2380, C(17,14) = 680, C(17,15) = 136,
    # C(17,16) = 17, C(17,17) = 1
    # Total ~ 89846, very manageable
    
    best_count = 0
    best_vertices = None
    
    for size in range(min_v, n + 1):
        for combo in combinations(range(n), size):
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
            
            vset = set(combo)
            degrees = []
            for v in combo:
                # Count neighbors of v within the induced subgraph
                deg = 0
                mask = adj_mask[v]
                # Count bits in mask that are also in vset
                # Use bit operations
                vmask = 0
                for u in combo:
                    vmask |= (1 << u)
                # Actually, let's just iterate since it's small
                for u in combo:
                    if u != v and (adj_mask[v] >> u) & 1:
                        deg += 1
                degrees.append(deg)
            
            distinct = len(set(degrees))
            if distinct > best_count:
                best_count = distinct
                best_vertices = list(combo)
        
        if time.time() - start_time > deadline:
            break
    
    if best_vertices is None:
        best_vertices = list(range(min_v))
    
    # Ensure we return Python ints in a list
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
