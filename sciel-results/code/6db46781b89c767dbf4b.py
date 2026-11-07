"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _count_distinct_degrees(vertices, adj):
    """Count distinct degrees in the induced subgraph on the given vertices."""
    vset = set(vertices)
    degrees = []
    for v in vertices:
        deg = sum(1 for u in vset if u in adj[v])
        degrees.append(deg)
    return len(set(degrees))


def _exhaustive_best():
    """Exhaustively search all subsets of size >= 8 to find the maximum distinct degrees."""
    n = P['n']
    min_v = P['minimum_vertices']
    adj = _build_adjacency()
    
    best_count = 0
    best_vertices = None
    
    # Enumerate all subsets of size from min_v to n
    for size in range(min_v, n + 1):
        for combo in combinations(range(n), size):
            count = _count_distinct_degrees(combo, adj)
            if count > best_count:
                best_count = count
                best_vertices = list(combo)
    
    return best_vertices if best_vertices else list(range(min_v))


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s
    
    # Exhaustive search is fast enough for n=17 with min_v=8
    # C(17,8) + C(17,9) + ... + C(17,17) is manageable
    best_vertices = _exhaustive_best()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to a simple valid answer if somehow we timed out
        return list(range(P['minimum_vertices']))
    
    # Ensure we return a list of Python ints
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
