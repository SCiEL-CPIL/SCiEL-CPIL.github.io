"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = P['n']
    
    # Build neighbor bitmasks
    adj_bits = [0] * n
    for u, v in P['edges']:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    # Precompute popcount for all 2^17 masks using DP
    # popcount[m] = number of 1-bits in m
    max_mask = 1 << n
    popcount = [0] * max_mask
    for m in range(1, max_mask):
        popcount[m] = popcount[m & (m - 1)] + 1
    
    best_score = 0
    best_vertices = None
    
    # Enumerate all subsets with at least 8 vertices
    # For each mask, compute distinct degrees in induced subgraph
    # Degree of vertex v in mask = popcount(adj_bits[v] & mask) - (1 if v in mask else 0)
    # Actually: degree = popcount(adj_bits[v] & mask) since v's own bit isn't in adj_bits[v]
    
    # We want to maximize distinct degrees, so try larger subsets first
    # but also check all sizes >= 8
    
    # Strategy: iterate masks from largest to smallest popcount
    # Group masks by popcount for efficiency
    
    # Actually, let's just iterate all masks and filter by popcount >= 8
    # 2^17 = 131072 masks, each requiring O(n) work = ~2.2M operations total
    
    # For speed, we can use a different approach:
    # For each mask, compute the set of degrees
    
    # Precompute which vertices are in each mask? No, that's too much memory.
    # Instead, for each mask, iterate through set bits.
    
    # Better: for each mask, compute degrees for all n vertices (only those in mask matter)
    # Use bit operations
    
    # Let's iterate masks and check popcount >= 8
    # To find the best, we can try to find masks with many distinct degrees
    
    # Key insight: max distinct degrees in k-vertex induced subgraph is at most k
    # (degrees range from 0 to k-1)
    # So for k=17, max is 17; for k=10, max is 10; etc.
    
    # Let's try to find a 10-vertex subset with 10 distinct degrees
    # Or larger subsets with even more distinct degrees
    
    # Iterate over all masks with popcount between 8 and 17
    # For each, compute distinct degrees
    
    # Optimization: only consider masks where popcount >= 8
    # Use Gray code or just iterate all masks
    
    # Simple approach: iterate all 2^17 masks
    found = False
    for mask in range(max_mask):
        if time.time() > deadline:
            break
        
        pc = popcount[mask]
        if pc < 8:
            continue
        
        # Compute degrees of vertices in this induced subgraph
        # For each vertex v in mask, degree = popcount(adj_bits[v] & mask)
        # Collect distinct degrees
        
        # Quick check: if pc <= best_score, can't improve
        if pc <= best_score:
            continue
        
        # Compute distinct degrees
        deg_set = set()
        m = mask
        v = 0
        while m:
            if m & 1:
                d = popcount[adj_bits[v] & mask]
                deg_set.add(d)
                if len(deg_set) > best_score:
                    # Could potentially improve
                    pass
            v += 1
            m >>= 1
        
        distinct = len(deg_set)
        if distinct > best_score:
            best_score = distinct
            best_vertices = [i for i in range(n) if mask & (1 << i)]
            if best_score == pc:
                # Can't do better than number of vertices
                # But we might find a larger subset with same or more distinct degrees
                # Actually, if we have k vertices with k distinct degrees, that's max for k
                # But a larger subset could have more distinct degrees
                pass
    
    # Ensure we have a valid result
    if best_vertices is None or len(best_vertices) < 8:
        # Fallback: first 8 vertices
        best_vertices = list(range(8))
    
    # Ensure all elements are Python ints
    result = [int(v) for v in best_vertices]
    return result
# EVOLVE-BLOCK-END
