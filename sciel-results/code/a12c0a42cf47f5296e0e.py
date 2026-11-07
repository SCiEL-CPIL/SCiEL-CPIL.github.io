# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute adjacency bitmasks for fast subset operations
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def count_distinct_degrees(vertices):
        """Count distinct degrees in the induced subgraph on 'vertices'."""
        vset = set(vertices)
        degrees = []
        for v in vertices:
            deg = sum(1 for u in vertices if u != v and u in adj[v])
            degrees.append(deg)
        return len(set(degrees))
    
    # We need to select at least min_vertices (8) distinct vertices from 17
    # Maximize the number of distinct degrees in the induced subgraph
    
    # Strategy: exhaustive search over subsets of size 8 to 17
    # C(17,8) = 24310, C(17,9) = 24310, C(17,10) = 19448, ...
    # Total subsets of size >= 8: sum_{k=8}^{17} C(17,k) = 2^17 - sum_{k=0}^{7} C(17,k)
    # sum_{k=0}^{7} C(17,k) = 1+17+136+680+2380+6188+12376+19448 = 41226
    # 2^17 = 131072
    # So about 89846 subsets total - feasible with optimization
    
    best_score = 0
    best_vertices = None
    
    # Precompute all subsets as bitmasks for sizes 8 to 17
    # Use bitmask representation for efficiency
    
    # For each subset, compute degrees quickly
    # Degree of vertex v in subset S = popcount(adj_mask[v] & S)
    
    all_vertices = list(range(n))
    
    # Iterate over all subsets of size >= min_vertices
    for size in range(min_vertices, n + 1):
        if time.time() > deadline:
            break
        
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            
            # Compute degrees using bitmask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degrees = []
            for v in subset:
                deg = popcount(adj_mask[v] & mask)
                degrees.append(deg)
            
            distinct = len(set(degrees))
            
            if distinct > best_score:
                best_score = distinct
                best_vertices = list(subset)
                
                # The maximum possible distinct degrees in a graph of size k is k
                # (all degrees distinct: 0,1,...,k-1) but that's impossible in simple graphs
                # Max distinct degrees is at most k-1 (since degree 0 and k-1 can't coexist unless k=1)
                # Actually max is k-1 for k>=2
                if best_score >= size - 1:
                    # Can't do better than this for this size, but larger sizes might allow more
                    # Actually we want max distinct degrees overall, so keep going
                    pass
    
    # If no valid result found (shouldn't happen), return a default
    if best_vertices is None:
        best_vertices = list(range(min_vertices))
    
    # Ensure we return Python ints in a list
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
