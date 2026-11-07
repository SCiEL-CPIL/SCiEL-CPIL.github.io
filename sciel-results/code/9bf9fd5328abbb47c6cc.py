"""Exhaustive search over all subsets of size >= 8 to maximize distinct induced degrees."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adjacency(n, edges):
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    return adj

def _count_distinct_degrees(mask, n, adj):
    """Count distinct degrees in the induced subgraph on vertices in mask."""
    # Build degree for each vertex in mask
    degrees = set()
    m = mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        # degree of v within mask = popcount(adj[v] & mask)
        deg = bin(adj[v] & mask).count('1')
        degrees.add(deg)
        m ^= lb
    return len(degrees)

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    adj = _build_adjacency(n, edges)
    
    # Precompute popcount for all masks (optional speedup, but bin().count is fine for 17 bits)
    # Actually let's precompute a popcount table for speed
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    best_count = 0
    best_mask = 0
    
    total_masks = 1 << n
    for mask in range(total_masks):
        if time.time() > deadline:
            break
        # Count vertices in mask
        cnt = popcount[mask]
        if cnt < min_verts:
            continue
        # Compute distinct degrees
        degs = set()
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            degs.add(popcount[adj[v] & mask])
            m ^= lb
        nd = len(degs)
        if nd > best_count:
            best_count = nd
            best_mask = mask
        # Early termination: max possible distinct degrees for cnt vertices is cnt (degrees 0..cnt-1)
        # If we've already found cnt distinct degrees, we can't do better for this size
        # But we still need to check larger subsets... actually max overall is min(cnt, 8) for 8+ verts
        # The absolute max is 8 (for 8-vertex subgraphs) or up to cnt for larger, but degrees max at cnt-1
        # So max distinct degrees = min(cnt, cnt) = cnt, but actually max degree in cnt-vertex graph is cnt-1
        # So distinct degrees <= cnt (values 0 to cnt-1)
        # For cnt >= 8, max is cnt, but we can stop if best_count == 8 since that's the theoretical max for 8-vertex
        # Actually for 9-vertex, max distinct degrees is 9 (0-8), etc. So no early stop at 8.
        # But 8 is a very strong bound. Let's continue.
    
    # Extract vertices from best_mask
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        result.append(int(v))
        m ^= lb
    
    # Ensure at least min_verts (should be guaranteed)
    if len(result) < min_verts:
        # Fallback: first min_verts vertices
        result = list(range(min_verts))
    
    return list(result)

# EVOLVE-BLOCK-END
