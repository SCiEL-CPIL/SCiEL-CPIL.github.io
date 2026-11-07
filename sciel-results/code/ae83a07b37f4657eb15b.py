"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute: for any subset, we need degrees within the induced subgraph
    # Strategy: enumerate subsets of size 8 to n, compute distinct degree count
    
    best_degrees = 0
    best_subset = None
    best_result = list(range(9))  # fallback
    
    # For efficiency, try all subsets of size 8 first (smallest valid),
    # then sizes 9, 10, ..., up to deadline
    # With 17 vertices, C(17,8) = 24310, C(17,9)=11628, etc.
    # Total subsets of size >=8: sum_{k=8}^{17} C(17,k) = 2^17 - sum_{k=0}^{7} C(17,k)
    # sum_{k=0}^{7} C(17,k) = 1+17+136+680+2380+6188+12376+19448 = 41226
    # 2^17 = 131072, so total = 131072 - 41226 = 89846 subsets
    
    # This is manageable within 85 seconds
    
    # We'll iterate over all subsets of size >= min_verts
    # For each, compute the induced subgraph degrees and count distinct values
    
    from itertools import combinations as comb
    
    # Precompute bitmasks for adjacency for speed
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def count_distinct_degrees(subset_mask):
        """Count distinct degrees in induced subgraph given as bitmask."""
        degrees = []
        for i in range(n):
            if subset_mask & (1 << i):
                deg = (adj_mask[i] & subset_mask).bit_count()
                degrees.append(deg)
        return len(set(degrees))
    
    # Enumerate all valid subsets
    for k in range(min_verts, n + 1):
        if time.time() > deadline - 2.0:
            break
        for subset in comb(range(n), k):
            if time.time() > deadline - 2.0:
                break
            
            # Build bitmask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            nd = count_distinct_degrees(mask)
            if nd > best_degrees:
                best_degrees = nd
                best_subset = subset
                best_result = list(subset)
    
    if best_subset is not None:
        return list(best_subset)
    return best_result

# EVOLVE-BLOCK-END
