# EVOLVE-BLOCK-START
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 85.0  # safety margin before 90s kill

    n = 16
    # The graph must be triangle-free to ensure every edge is essential.
    # Proof: If a triangle (u, v, w) exists, removing edge (u, v) leaves path u-w-v of length 2.
    # Thus dist(u, v) remains 2, so diameter does not increase above 2, violating the "essential" condition.
    # Mantel's Theorem states max edges in a triangle-free graph on n vertices is floor(n^2/4).
    # For n=16, max edges = 64, achieved by K_{8,8}.
    
    # Construct K_{8,8}
    # Partition A: 0..7, Partition B: 8..15
    # Edge (u, v) exists iff u in A and v in B.
    
    edges = [[u, v] for u in range(8) for v in range(8, 16)]
    
    # Deadline guard
    if time.time() > _deadline:
        return edges
    
    return edges
# EVOLVE-BLOCK-END
