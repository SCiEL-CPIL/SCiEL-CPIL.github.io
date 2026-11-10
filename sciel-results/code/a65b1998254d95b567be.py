"""Search for a small 3-regular graph of girth >= g, using known cages and LUW construction."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges).
    Optimized for integer IDs."""
    q = k
    # d must be odd for the girth formula d+5 to hold cleanly?
    # LUW CD(d,q) has girth >= d+5 if d is odd.
    # We need girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # Pick smallest odd d >= g - 5.
    d0 = g - 5
    if d0 < 1:
        d0 = 1
    if d0 % 2 == 0:
        d0 += 1
    d = d0
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Precompute powers of q for ID decoding/encoding
    # ID = side * q^d + x_1 * q^(d-1) + ... + x_d * q^0
    q_pow = [1] * (d + 1)
    for i in range(1, d + 1):
        q_pow[i] = q_pow[i-1] * q
        
    n = 2 * q_pow[d]
    edges = []
    
    # We only need to generate edges from side 0 to side 1 to avoid duplicates
    # Iterate over all vertices on side 0
    for v in range(q_pow[d]):
        # Decode v into x_1..x_d
        # v = x_1 * q^(d-1) + ... + x_d
        x = [0] * (d + 1)
        temp = v
        for i in range(d, 0, -1):
            x[i] = temp % q
            temp //= q
        # x[1]..x[d] are the coordinates
        
        for first in range(q):
            # y[1] = first
            # y[c] for c=2..d
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b]))
            
            # Encode y into u
            u = 0
            for i in range(1, d + 1):
                u = u * q + y[i]
            
            # Edge between v (side 0) and q^d + u (side 1)
            edges.append((v, q_pow[d] + u))
            
    return n, edges

def _get_small_cage(k, g):
    """Return hardcoded small cages for k=3."""
    if k != 3:
        return None
    
    if g <= 4:
        # K3,3
        edges = [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)]
        return 6, edges
    elif g <= 5:
        # Petersen graph
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5), (1, 6),
            (2, 7), (2, 8), (2, 9),
            (3, 4), (3, 5), (3, 6),
            (4, 7), (4, 8), (4, 9),
            (5, 7), (5, 8), (5, 9),
            (6, 7), (6, 8), (6, 9)
        ]
        # Wait, Petersen is (0-1-2-3-4-0) outer, (5-6-7-8-9-5) inner?
        # Standard Petersen:
        # Outer: 0-1-2-3-4-0
        # Inner: 5-6-7-8-9-5 (star)
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        # Let's use a reliable Petersen edge list
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer cycle
            (5, 6), (6, 7), (7, 8), (8, 9), (9, 5), # Inner cycle? No, Petersen inner is a star K_{1,4}
            # Correct Petersen:
            # Vertices 0..4 form a 5-cycle. Vertices 5..9 form a 5-cycle? No.
            # Vertices 0..4 are outer. 5..9 are inner.
            # Inner edges: (5,7), (7,9), (9,6), (6,8), (8,5).
            # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9).
            # Outer: (0,1), (1,2), (2,3), (3,4), (4,0).
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        return 10, edges
    elif g <= 6:
        # (3,6)-cage, 14 vertices
        # This is the graph formed by a 14-cycle with chords?
        # Actually, the (3,6)-cage is the "Tutte graph" minus something?
        # It is the graph with 14 vertices.
        # Edges:
        # 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-10, 10-11, 11-12, 12-13, 13-0 (Cycle 14)
        # Plus chords: 0-7, 1-8, 2-9, 3-10, 4-11, 5-12, 6-13?
        # This creates triangles? 0-1-8-0? No, 8 is not connected to 0.
        # 0-1, 1-8, 8-? 
        # Let's use a known (3,6)-cage.
        # It is the graph obtained from K3,3 by subdividing edges? No.
        # It is the "Möbius-Kantor" is 16.
        # The (3,6)-cage is the 14-vertex graph.
        # Edges:
        # (0,1), (0,2), (0,3)
        # (1,4), (1,5), (1,6)
        # (2,7), (2,8), (2,9)
        # (3,10), (3,11), (3,12)
        # (4,13), (4,5), (4,6) -- Wait, 5 and 6 are already used.
        # This is getting risky. I will skip hardcoding g=6 if I'm not sure.
        # LUW(d=3) gives n=54.
        # If I return n=54 for g=6, it's valid but not optimal.
        # I'll stick to g=4,5 hardcoded.
        return None
    else:
        return None

def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 145.0

    # Try small hardcoded cages
    small = _get_small_cage(k, g)
    if small:
        return small

    # Fallback: LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        # Last resort: raise
        raise

# EVOLVE-BLOCK-END
