"""Exhaustive bitmask search for maximum distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _popcount(x):
    """Fast popcount using bit manipulation."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def _distinct_degrees_mask(mask, adj, n):
    """Compute number of distinct degrees in induced subgraph given as bitmask."""
    degs = set()
    for v in range(n):
        if mask & (1 << v):
            degs.add(_popcount(adj[v] & mask))
    return len(degs)


def _mask_to_list(mask, n):
    """Convert bitmask to list of vertex indices."""
    return [v for v in range(n) if mask & (1 << v)]


def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Build adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    best_score = -1
    best_mask = 0

    # Enumerate all subsets of size >= minv, from largest to smallest
    # For each size, the max possible distinct degrees is `size`
    for size in range(n, minv - 1, -1):
        if best_score >= size:
            break

        # Generate all combinations of `size` vertices from range(n)
        # Use iterative combinatorial generation for efficiency
        # We'll use a simple recursive/iterative approach
        vertices = list(range(n))
        
        # Use itertools for clean combination generation
        import itertools
        for combo in itertools.combinations(vertices, size):
            if time.time() > deadline:
                break
            
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            score = _distinct_degrees_mask(mask, adj, n)
            
            if score > best_score:
                best_score = score
                best_mask = mask
                
                # If we hit theoretical max for this size, can't do better
                if score == size:
                    return [int(v) for v in _mask_to_list(best_mask, n)]
        
        if time.time() > deadline:
            break

    # Fallback
    if best_mask == 0:
        best_mask = (1 << minv) - 1  # vertices 0..minv-1

    return [int(v) for v in _mask_to_list(best_mask, n)]
# EVOLVE-BLOCK-END
